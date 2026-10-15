use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct AwaySnap {
    pub name: String,
    pub detail: String,
    pub alarm: bool,
}

struct Clock {
    name: String,
    seen_s: f32,
    away_s: f32,
    logged: bool,
}

pub struct Watch {
    path: PathBuf,
    clocks: HashMap<String, Clock>,
    today: String,
    today_s: HashMap<String, f32>,
    limit_s: f32,
}

impl Watch {
    pub fn open(path: &Path) -> Self {
        let today = today_string();
        let today_s = load_today(path, &today);
        Self {
            path: path.to_path_buf(),
            clocks: HashMap::new(),
            today,
            today_s,
            limit_s: 30.0,
        }
    }

    pub fn tick(&mut self, present: &[String], roster: &[(String, String)], dt: f32, on_duty: bool) -> Vec<AwaySnap> {
        let today = today_string();
        if today != self.today {
            self.today = today;
            self.today_s = load_today(&self.path, &self.today);
            for clock in self.clocks.values_mut() {
                clock.away_s = 0.0;
                clock.seen_s = 0.0;
                clock.logged = false;
            }
        }
        let dt = dt.clamp(0.0, 1.0);
        for (id, name) in roster {
            let clock = self.clocks.entry(id.clone()).or_insert_with(|| Clock {
                name: name.clone(),
                seen_s: 0.0,
                away_s: 0.0,
                logged: false,
            });
            clock.name = name.clone();
            if present.iter().any(|item| item == id) {
                if clock.away_s >= 15.0 && on_duty {
                    let duration = clock.away_s;
                    *self.today_s.entry(id.clone()).or_insert(0.0) += duration;
                    let _ = append_away(&self.path, id, &clock.name, duration, &self.today);
                }
                clock.away_s = 0.0;
                clock.logged = false;
                clock.seen_s += dt;
            } else if clock.seen_s >= 6.0 {
                clock.away_s += dt;
                if on_duty && clock.away_s >= self.limit_s && !clock.logged {
                    clock.logged = true;
                }
            } else {
                clock.seen_s = 0.0;
                clock.away_s = 0.0;
            }
        }
        let mut rows: Vec<AwaySnap> = roster
            .iter()
            .map(|(id, name)| {
                let clock = self.clocks.get(id);
                let away = clock.map(|clock| clock.away_s).unwrap_or(0.0);
                let seen = clock.map(|clock| clock.seen_s).unwrap_or(0.0);
                let today = self.today_s.get(id).copied().unwrap_or(0.0);
                let (detail, alarm) = if away >= self.limit_s {
                    (format!("离岗 {} · 今日 {}", duration_label(away), duration_label(today)), true)
                } else if away > 0.0 {
                    let note = if on_duty { "" } else { " · 现在不计" };
                    (format!("暂离 {}{note} · 今日 {}", duration_label(away), duration_label(today)), false)
                } else if seen > 0.0 || present.iter().any(|item| item == id) {
                    (format!("在岗 · 今日 {}", duration_label(today)), false)
                } else {
                    (format!("未到 · 今日 {}", duration_label(today)), false)
                };
                AwaySnap {
                    name: name.clone(),
                    detail,
                    alarm,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.alarm.cmp(&a.alarm).then_with(|| a.name.cmp(&b.name)));
        rows
    }

    pub fn on_duty_now() -> bool {
        let Some((weekday, minute)) = local_minutes() else {
            return true;
        };
        (1..=5).contains(&weekday) && (9 * 60..18 * 60 + 30).contains(&minute)
    }
}

fn duration_label(secs: f32) -> String {
    let total = secs.round().max(0.0) as u32;
    if total < 60 {
        format!("{total} 秒")
    } else if total < 3600 {
        format!("{} 分 {} 秒", total / 60, total % 60)
    } else {
        format!("{} 小时 {} 分", total / 3600, (total % 3600) / 60)
    }
}

fn today_string() -> String {
    let Some((_, _, ymd)) = local_parts() else {
        return String::new();
    };
    ymd
}

fn local_minutes() -> Option<(u32, u32)> {
    let (weekday, minute, _) = local_parts()?;
    Some((weekday, minute))
}

fn local_parts() -> Option<(u32, u32, String)> {
    let raw = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    let local = raw + 8 * 3600;
    let days = local.div_euclid(86_400);
    let tod = local.rem_euclid(86_400) as u32;
    let weekday = ((days + 3).rem_euclid(7) + 1) as u32;
    let minute = tod / 60;
    let year_day = days;
    let (year, month, day) = civil_from_days(year_day);
    Some((weekday, minute, format!("{year:04}-{month:02}-{day:02}")))
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month as u32, day as u32)
}

fn load_today(path: &Path, today: &str) -> HashMap<String, f32> {
    let mut totals = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return totals;
    };
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("date").and_then(|item| item.as_str()) != Some(today) {
            continue;
        }
        let Some(id) = value.get("person_id").and_then(|item| item.as_str()) else {
            continue;
        };
        let duration = value.get("duration_s").and_then(|item| item.as_f64()).unwrap_or(0.0) as f32;
        if duration > 0.0 {
            *totals.entry(id.to_string()).or_insert(0.0) += duration;
        }
    }
    totals
}

fn append_away(path: &Path, id: &str, name: &str, duration: f32, today: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|item| item.as_secs_f64()).unwrap_or(0.0);
    let start = now - f64::from(duration);
    let line = serde_json::json!({
        "event": "away",
        "person_id": id,
        "name": name,
        "names": [name],
        "t": now,
        "start": start,
        "end": now,
        "duration_s": (duration * 10.0).round() / 10.0,
        "duration": duration_label(duration),
        "alarm": duration >= 30.0,
        "date": today
    });
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")?;
    Ok(())
}
