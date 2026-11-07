//! 本机实时指标：内存、显存、GPU、功耗（RAPL Energy Meter）、本地模型（llama-server）和 ComfyUI。
//!
//! 后台线程每秒采样一次，结果放在全局快照里；主题用 `pc.*` 变量读取，
//! 悬停提示用 [`summary_lines`]。只有主题里用到 `pc.` 变量时才会启动采样线程。

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 用户可改的设置，存放在 `%APPDATA%\ClaudeCodeUsageMonitor\local-metrics.json`。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalConfig {
    /// 本地模型（llama-server）的地址，Ornith 默认 8090。
    pub llm_url: String,
    /// 任务栏上显示的本地模型名字。
    pub llm_name: String,
    /// ComfyUI 的地址。
    pub comfy_url: String,
    /// 功耗进度条的满格功率（瓦）。8745HS 默认 54 W。
    pub power_limit_w: f64,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            llm_url: "http://127.0.0.1:8090".into(),
            llm_name: "Ornith".into(),
            comfy_url: "http://127.0.0.1:8188".into(),
            power_limit_w: 54.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum LlmStatus {
    #[default]
    Offline,
    Loading,
    Idle,
    Generating,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub mem_total: f64,
    pub mem_used: f64,
    /// 占内存最多的程序：(名字, 字节)，已经把共享显存算进去。
    pub top_mem: Vec<(String, f64)>,
    pub gpu_util: Option<f64>,
    pub gpu_dedicated: Option<f64>,
    pub gpu_shared: Option<f64>,
    pub pkg_watts: Option<f64>,
    pub core_watts: Option<f64>,
    pub on_ac: Option<bool>,
    pub battery_pct: Option<f64>,
    pub llm_status: LlmStatus,
    pub llm_tps: Option<f64>,
    pub llm_ctx_pct: Option<f64>,
    pub comfy_online: bool,
    pub comfy_running: u32,
    pub comfy_pending: u32,
    pub comfy_elapsed: Option<Duration>,
}

static STARTED: AtomicBool = AtomicBool::new(false);
static SNAPSHOT: Mutex<Option<Snapshot>> = Mutex::new(None);

pub fn config() -> &'static LocalConfig {
    static CONFIG: OnceLock<LocalConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let path = crate::app_settings::app_data_directory().join("local-metrics.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap_or_else(|error| {
                crate::diagnose::log(format!("local-metrics.json invalid, using defaults: {error}"));
                LocalConfig::default()
            }),
            Err(_) => {
                let config = LocalConfig::default();
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Ok(text) = serde_json::to_string_pretty(&config) {
                    let _ = std::fs::write(&path, text);
                }
                config
            }
        }
    })
}

/// 启动后台采样线程（只启动一次）。
pub fn ensure_started() {
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("local-metrics".into())
        .spawn(sampler_thread);
}

pub fn is_running() -> bool {
    STARTED.load(Ordering::SeqCst)
}

pub fn snapshot() -> Option<Snapshot> {
    SNAPSHOT.lock().ok().and_then(|guard| guard.clone())
}

fn sampler_thread() {
    let mut sampler = Sampler::new();
    loop {
        let started = Instant::now();
        let snap = sampler.sample();
        if let Ok(mut guard) = SNAPSHOT.lock() {
            *guard = Some(snap);
        }
        let spent = started.elapsed();
        std::thread::sleep(SAMPLE_INTERVAL.saturating_sub(spent).max(Duration::from_millis(200)));
    }
}

// ---------------------------------------------------------------------------
// 采样
// ---------------------------------------------------------------------------

struct Sampler {
    pdh: Option<pdh::Query>,
    agent: ureq::Agent,
    last_decoded: Option<(i64, f64, Instant)>,
    tps_smooth: Option<f64>,
    last_tps: Option<f64>,
    comfy_since: Option<Instant>,
    process_names: BTreeMap<u32, String>,
    process_names_at: Option<Instant>,
}

impl Sampler {
    fn new() -> Self {
        let pdh = pdh::Query::open();
        if pdh.is_none() {
            crate::diagnose::log("local metrics: PDH query unavailable");
        }
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_millis(800)))
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            pdh,
            agent,
            last_decoded: None,
            tps_smooth: None,
            last_tps: None,
            comfy_since: None,
            process_names: BTreeMap::new(),
            process_names_at: None,
        }
    }

    fn sample(&mut self) -> Snapshot {
        let mut snap = Snapshot::default();
        read_memory(&mut snap);
        read_power_status(&mut snap);
        if let Some(query) = self.pdh.as_mut() {
            query.collect();
            // GPU 利用率：按引擎类型（3D、Compute 等）分别求和，取最大的那一类。
            let mut by_engine: BTreeMap<String, f64> = BTreeMap::new();
            for (name, value) in query.values(pdh::GPU_ENGINE) {
                let engine = name
                    .split("engtype_")
                    .nth(1)
                    .unwrap_or("other")
                    .to_string();
                *by_engine.entry(engine).or_default() += value;
            }
            snap.gpu_util = by_engine.values().cloned().fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            });
            if let Some(v) = snap.gpu_util.as_mut() {
                *v = v.clamp(0.0, 100.0);
            }
            let max_of = |values: Vec<(String, f64)>| {
                values.into_iter().map(|(_, v)| v).fold(None, |acc: Option<f64>, v| {
                    Some(acc.map_or(v, |a| a.max(v)))
                })
            };
            snap.gpu_dedicated = max_of(query.values(pdh::GPU_DEDICATED));
            snap.gpu_shared = max_of(query.values(pdh::GPU_SHARED));

            // 功耗：Energy Meter 的 Power 计数器（单位一般是毫瓦）。
            let mut core_total = 0.0;
            let mut core_seen = false;
            for (name, value) in query.values(pdh::ENERGY_POWER) {
                let watts = normalize_watts(value);
                let lower = name.to_ascii_lowercase();
                if lower.contains("pkg") || lower.contains("package0_pkg") {
                    snap.pkg_watts = Some(watts);
                } else if lower.contains("_core") {
                    core_total += watts;
                    core_seen = true;
                }
            }
            if core_seen {
                snap.core_watts = Some(core_total);
            }

            // 每个程序的内存：私有工作集 + 显卡占用（核显的显存来自系统内存）。
            let mut per_app: BTreeMap<String, f64> = BTreeMap::new();
            for (name, value) in query.values(pdh::PROCESS_PRIVATE) {
                if name == "_Total" || name == "Idle" || name.is_empty() {
                    continue;
                }
                let base = name.split('#').next().unwrap_or(&name).to_string();
                *per_app.entry(friendly_name(&base)).or_default() += value;
            }
            let gpu_proc = query.values(pdh::GPU_PROCESS_SHARED);
            if !gpu_proc.is_empty() {
                self.refresh_process_names();
                for (name, value) in gpu_proc {
                    if value < 64.0 * 1024.0 * 1024.0 {
                        continue;
                    }
                    let pid = name
                        .strip_prefix("pid_")
                        .and_then(|rest| rest.split('_').next())
                        .and_then(|pid| pid.parse::<u32>().ok());
                    if let Some(exe) = pid.and_then(|pid| self.process_names.get(&pid)) {
                        *per_app.entry(friendly_name(exe)).or_default() += value;
                    }
                }
            }
            let mut apps: Vec<(String, f64)> = per_app.into_iter().collect();
            apps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            apps.truncate(3);
            snap.top_mem = apps;
        }
        self.sample_llm(&mut snap);
        self.sample_comfy(&mut snap);
        snap
    }

    fn refresh_process_names(&mut self) {
        if self
            .process_names_at
            .is_some_and(|at| at.elapsed() < Duration::from_secs(10))
        {
            return;
        }
        self.process_names = process_names();
        self.process_names_at = Some(Instant::now());
    }

    fn get_json(&self, url: &str) -> Result<(u16, serde_json::Value), ()> {
        let mut response = self.agent.get(url).call().map_err(|_| ())?;
        let status = response.status().as_u16();
        let value = response
            .body_mut()
            .read_json::<serde_json::Value>()
            .unwrap_or(serde_json::Value::Null);
        Ok((status, value))
    }

    fn sample_llm(&mut self, snap: &mut Snapshot) {
        let base = config().llm_url.trim_end_matches('/').to_string();
        match self.get_json(&format!("{base}/health")) {
            Err(()) => {
                snap.llm_status = LlmStatus::Offline;
                self.last_decoded = None;
                self.tps_smooth = None;
                return;
            }
            Ok((503, _)) => {
                snap.llm_status = LlmStatus::Loading;
                return;
            }
            Ok(_) => snap.llm_status = LlmStatus::Idle,
        }
        let Ok((200, serde_json::Value::Array(slots))) = self.get_json(&format!("{base}/slots"))
        else {
            snap.llm_tps = self.last_tps;
            return;
        };
        let mut busy = false;
        let mut decoded_total: i64 = 0;
        let mut task_sum: f64 = 0.0;
        let mut ctx_pct: Option<f64> = None;
        for slot in &slots {
            if slot.get("is_processing").and_then(|v| v.as_bool()) == Some(true) {
                busy = true;
                decoded_total += find_number(slot, "n_decoded").unwrap_or(0.0) as i64;
                task_sum += slot.get("id_task").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let n_ctx = slot.get("n_ctx").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let used = find_number(slot, "n_past")
                    .or_else(|| find_number(slot, "n_prompt_tokens_processed"))
                    .unwrap_or(0.0);
                if n_ctx > 0.0 && used > 0.0 {
                    ctx_pct = Some((used / n_ctx * 100.0).clamp(0.0, 100.0));
                }
            }
        }
        snap.llm_ctx_pct = ctx_pct;
        if busy {
            snap.llm_status = LlmStatus::Generating;
            let now = Instant::now();
            if let Some((prev, prev_task, at)) = self.last_decoded {
                let dt = now.duration_since(at).as_secs_f64();
                if prev_task == task_sum && decoded_total >= prev && dt > 0.2 {
                    let rate = (decoded_total - prev) as f64 / dt;
                    if rate > 0.0 {
                        let smooth = match self.tps_smooth {
                            Some(old) => old * 0.5 + rate * 0.5,
                            None => rate,
                        };
                        self.tps_smooth = Some(smooth);
                        self.last_tps = Some(smooth);
                    }
                }
            }
            self.last_decoded = Some((decoded_total, task_sum, now));
            snap.llm_tps = self.tps_smooth;
        } else {
            self.last_decoded = None;
            self.tps_smooth = None;
            snap.llm_tps = self.last_tps;
        }
    }

    fn sample_comfy(&mut self, snap: &mut Snapshot) {
        let base = config().comfy_url.trim_end_matches('/').to_string();
        match self.get_json(&format!("{base}/queue")) {
            Ok((200, value)) => {
                snap.comfy_online = true;
                let count = |key: &str| {
                    value
                        .get(key)
                        .and_then(|v| v.as_array())
                        .map(|a| a.len() as u32)
                        .unwrap_or(0)
                };
                snap.comfy_running = count("queue_running");
                snap.comfy_pending = count("queue_pending");
                if snap.comfy_running > 0 {
                    let since = *self.comfy_since.get_or_insert_with(Instant::now);
                    snap.comfy_elapsed = Some(since.elapsed());
                } else {
                    self.comfy_since = None;
                }
            }
            _ => {
                snap.comfy_online = false;
                self.comfy_since = None;
            }
        }
    }
}

/// Energy Meter 的 Power 一般是毫瓦；笔记本处理器不可能低于 0.2 W 或高于 200 W，
/// 据此判断单位，避免读成 1000 倍。
fn normalize_watts(value: f64) -> f64 {
    if value >= 200.0 {
        value / 1000.0
    } else {
        value
    }
}

fn friendly_name(exe: &str) -> String {
    let base = exe.trim_end_matches(".exe").trim_end_matches(".EXE");
    let lower = base.to_ascii_lowercase();
    if lower.starts_with("llama-server") || lower.starts_with("ornith") {
        return config().llm_name.clone();
    }
    match lower.as_str() {
        "msedge" | "msedgewebview2" => "Edge".into(),
        "chrome" => "Chrome".into(),
        "firefox" => "Firefox".into(),
        "code" => "VS Code".into(),
        "claude" => "Claude".into(),
        "python" | "pythonw" => "Python".into(),
        _ => base.to_string(),
    }
}

fn find_number(value: &serde_json::Value, key: &str) -> Option<f64> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(found) = map.get(key).and_then(|v| v.as_f64()) {
                return Some(found);
            }
            map.values().find_map(|v| find_number(v, key))
        }
        serde_json::Value::Array(items) => items.iter().find_map(|v| find_number(v, key)),
        _ => None,
    }
}

fn read_memory(snap: &mut Snapshot) {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) }.is_ok() {
        snap.mem_total = status.ullTotalPhys as f64;
        snap.mem_used = status.ullTotalPhys.saturating_sub(status.ullAvailPhys) as f64;
    }
}

fn read_power_status(snap: &mut Snapshot) {
    use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut status = SYSTEM_POWER_STATUS::default();
    if unsafe { GetSystemPowerStatus(&mut status) }.is_ok() {
        snap.on_ac = match status.ACLineStatus {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };
        if status.BatteryLifePercent <= 100 && status.BatteryFlag != 128 {
            snap.battery_pct = Some(status.BatteryLifePercent as f64);
        }
    }
}

fn process_names() -> BTreeMap<u32, String> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    let mut names = BTreeMap::new();
    unsafe {
        let Ok(handle) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else {
            return names;
        };
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        if Process32FirstW(handle, &mut entry).is_ok() {
            loop {
                let len = entry
                    .szExeFile
                    .iter()
                    .position(|c| *c == 0)
                    .unwrap_or(entry.szExeFile.len());
                names.insert(
                    entry.th32ProcessID,
                    String::from_utf16_lossy(&entry.szExeFile[..len]),
                );
                if Process32NextW(handle, &mut entry).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(handle);
    }
    names
}

mod pdh {
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
        PDH_FMT, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA,
    };

    pub const GPU_ENGINE: usize = 0;
    pub const GPU_DEDICATED: usize = 1;
    pub const GPU_SHARED: usize = 2;
    pub const GPU_PROCESS_SHARED: usize = 3;
    pub const ENERGY_POWER: usize = 4;
    pub const PROCESS_PRIVATE: usize = 5;

    const PATHS: [&str; 6] = [
        r"\GPU Engine(*)\Utilization Percentage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        r"\GPU Adapter Memory(*)\Shared Usage",
        r"\GPU Process Memory(*)\Shared Usage",
        r"\Energy Meter(*)\Power",
        r"\Process(*)\Working Set - Private",
    ];

    /// PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
    const FORMAT: PDH_FMT = PDH_FMT(0x0000_0200 | 0x0000_8000);

    pub struct Query {
        query: PDH_HQUERY,
        counters: Vec<Option<PDH_HCOUNTER>>,
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    impl Query {
        pub fn open() -> Option<Self> {
            let mut query = PDH_HQUERY::default();
            let status = unsafe { PdhOpenQueryW(PCWSTR::null(), 0, &mut query) };
            if status != 0 {
                return None;
            }
            let counters = PATHS
                .iter()
                .map(|path| {
                    let path = wide(path);
                    let mut counter = PDH_HCOUNTER::default();
                    let status = unsafe {
                        PdhAddEnglishCounterW(query, PCWSTR(path.as_ptr()), 0, &mut counter)
                    };
                    (status == 0).then_some(counter)
                })
                .collect();
            Some(Self { query, counters })
        }

        pub fn collect(&mut self) {
            unsafe {
                let _ = PdhCollectQueryData(self.query);
            }
        }

        /// 读取一个通配计数器的所有实例：(实例名, 值)。
        pub fn values(&self, index: usize) -> Vec<(String, f64)> {
            let Some(Some(counter)) = self.counters.get(index).copied() else {
                return Vec::new();
            };
            let mut out = Vec::new();
            unsafe {
                let mut size: u32 = 0;
                let mut count: u32 = 0;
                let status =
                    PdhGetFormattedCounterArrayW(counter, FORMAT, &mut size, &mut count, None);
                if status != PDH_MORE_DATA || size == 0 {
                    return out;
                }
                // 缓冲区按 8 字节对齐分配，里面先是条目数组，后面是实例名字符串。
                let mut buffer = vec![0u64; (size as usize).div_ceil(8) + 1];
                let items = buffer.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
                let status = PdhGetFormattedCounterArrayW(
                    counter,
                    FORMAT,
                    &mut size,
                    &mut count,
                    Some(items),
                );
                if status != 0 {
                    return out;
                }
                for i in 0..count as usize {
                    let item = &*items.add(i);
                    if item.FmtValue.CStatus != 0 {
                        continue;
                    }
                    let name = pwstr_to_string(item.szName);
                    out.push((name, item.FmtValue.Anonymous.doubleValue));
                }
            }
            out
        }
    }

    unsafe fn pwstr_to_string(text: PWSTR) -> String {
        if text.is_null() {
            return String::new();
        }
        unsafe { text.to_string().unwrap_or_default() }
    }
}

// ---------------------------------------------------------------------------
// 给主题和悬停提示用的文字
// ---------------------------------------------------------------------------

fn gb(bytes: f64) -> String {
    let value = bytes / GIB;
    if value >= 10.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn level(pct: f64, warn: f64, crit: f64) -> f64 {
    if pct >= crit {
        2.0
    } else if pct >= warn {
        1.0
    } else {
        0.0
    }
}

/// 把快照写进主题变量。没有快照时也写入占位值，避免主题显示成 "--"。
pub fn insert_into(context: &mut crate::theme_engine::DataContext) {
    let snap = snapshot().unwrap_or_default();
    let config = config();

    let mem_pct = if snap.mem_total > 0.0 {
        snap.mem_used / snap.mem_total * 100.0
    } else {
        0.0
    };
    context.insert("pc.mem.percentage", mem_pct);
    context.insert("pc.mem.level", level(mem_pct, 80.0, 90.0));
    context.insert_string(
        "pc.mem.text",
        if snap.mem_total > 0.0 {
            format!("{}/{}G", gb(snap.mem_used), gb(snap.mem_total))
        } else {
            "--".into()
        },
    );

    let gpu = snap.gpu_util.unwrap_or(0.0);
    context.insert("pc.gpu.percentage", gpu);
    context.insert_string("pc.gpu.text", format!("{gpu:.0}%"));
    let vram = snap.gpu_dedicated.unwrap_or(0.0) + snap.gpu_shared.unwrap_or(0.0);
    context.insert_string("pc.vram.text", format!("{}G", gb(vram)));

    let watts = snap.pkg_watts.unwrap_or(0.0);
    let limit = config.power_limit_w.max(1.0);
    let power_pct = (watts / limit * 100.0).clamp(0.0, 100.0);
    context.insert("pc.power.available", snap.pkg_watts.is_some() as u8 as f64);
    context.insert("pc.power.percentage", power_pct);
    context.insert("pc.power.level", level(power_pct, 80.0, 95.0));
    context.insert_string(
        "pc.power.text",
        match snap.pkg_watts {
            Some(w) => format!("{w:.0} W"),
            None => "--".into(),
        },
    );

    context.insert_string("pc.llm.name", config.llm_name.as_str());
    let (llm_text, busy) = match snap.llm_status {
        LlmStatus::Offline => ("未运行".to_string(), 0.0),
        LlmStatus::Loading => ("加载中".to_string(), 0.0),
        LlmStatus::Idle => ("空闲".to_string(), 0.0),
        LlmStatus::Generating => (
            match snap.llm_tps {
                Some(tps) => format!("{tps:.1} t/s"),
                None => "生成中".to_string(),
            },
            1.0,
        ),
    };
    context.insert_string("pc.llm.text", llm_text);
    context.insert("pc.llm.busy", busy);
    context.insert(
        "pc.llm.online",
        (snap.llm_status != LlmStatus::Offline) as u8 as f64,
    );

    let comfy_active = snap.comfy_running > 0 || snap.comfy_pending > 0;
    context.insert("pc.comfy.active", comfy_active as u8 as f64);
    context.insert_string(
        "pc.comfy.text",
        if snap.comfy_running > 0 {
            let secs = snap.comfy_elapsed.map(|d| d.as_secs()).unwrap_or(0);
            if snap.comfy_pending > 0 {
                format!("{}s +{}", secs, snap.comfy_pending)
            } else {
                format!("{secs}s")
            }
        } else if snap.comfy_pending > 0 {
            format!("排队 {}", snap.comfy_pending)
        } else {
            String::new()
        },
    );
}

/// 悬停提示里追加的几行。
pub fn summary_lines() -> Vec<String> {
    if !is_running() {
        return Vec::new();
    }
    let Some(snap) = snapshot() else {
        return Vec::new();
    };
    let config = config();
    let mut lines = Vec::new();
    let llm = match snap.llm_status {
        LlmStatus::Offline => "未运行".to_string(),
        LlmStatus::Loading => "正在加载模型".to_string(),
        LlmStatus::Idle => match snap.llm_tps {
            Some(tps) => format!("空闲（上次 {tps:.1} tok/s）"),
            None => "空闲".to_string(),
        },
        LlmStatus::Generating => match snap.llm_tps {
            Some(tps) => format!("正在生成 {tps:.1} tok/s"),
            None => "正在生成".to_string(),
        },
    };
    let ctx = snap
        .llm_ctx_pct
        .map(|p| format!("，上下文已用 {p:.0}%"))
        .unwrap_or_default();
    lines.push(format!("{}：{llm}{ctx}", config.llm_name));
    if snap.comfy_online {
        let state = if snap.comfy_running > 0 {
            let secs = snap.comfy_elapsed.map(|d| d.as_secs()).unwrap_or(0);
            format!("正在生成（已 {secs} 秒），排队 {}", snap.comfy_pending)
        } else if snap.comfy_pending > 0 {
            format!("排队 {}", snap.comfy_pending)
        } else {
            "空闲".to_string()
        };
        lines.push(format!("ComfyUI：{state}"));
    }
    if snap.mem_total > 0.0 {
        let top = snap
            .top_mem
            .iter()
            .map(|(name, bytes)| format!("{name} {}G", gb(*bytes)))
            .collect::<Vec<_>>()
            .join("、");
        lines.push(format!(
            "内存：{}/{} GB{}",
            gb(snap.mem_used),
            gb(snap.mem_total),
            if top.is_empty() {
                String::new()
            } else {
                format!("（最多：{top}）")
            }
        ));
    }
    if snap.gpu_util.is_some() || snap.gpu_shared.is_some() {
        lines.push(format!(
            "显卡：{:.0}%，显存 专用 {}G + 共享 {}G",
            snap.gpu_util.unwrap_or(0.0),
            gb(snap.gpu_dedicated.unwrap_or(0.0)),
            gb(snap.gpu_shared.unwrap_or(0.0))
        ));
    }
    let mut power = Vec::new();
    if let Some(w) = snap.pkg_watts {
        power.push(format!("处理器 {w:.1} W"));
        if let Some(core) = snap.core_watts {
            power.push(format!("其中核心 {core:.1} W、核显等 {:.1} W", (w - core).max(0.0)));
        }
    }
    match (snap.on_ac, snap.battery_pct) {
        (Some(true), Some(p)) => power.push(format!("插电，电量 {p:.0}%")),
        (Some(false), Some(p)) => power.push(format!("用电池，电量 {p:.0}%")),
        _ => {}
    }
    if !power.is_empty() {
        lines.push(format!("功耗：{}", power.join("，")));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_monitor_theme_is_default_and_valid() {
        let path = crate::theme_engine::ensure_starter_theme().expect("built-in themes must be valid");
        assert!(path.ends_with("pc-agent-monitor.json"), "{}", path.display());
        let theme = crate::theme_engine::load_theme(&path).expect("theme loads");
        let errors = theme.validate();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            theme.current_time_refresh_interval(),
            Some(Duration::from_secs(2))
        );
    }

    #[test]
    fn pc_monitor_power_units() {
        assert!((normalize_watts(15_000.0) - 15.0).abs() < 1e-9);
        assert!((normalize_watts(15.0) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn pc_monitor_finds_nested_numbers() {
        let slot = serde_json::json!({"id": 0, "next_token": [{"n_decoded": 42}]});
        assert_eq!(find_number(&slot, "n_decoded"), Some(42.0));
    }
}
