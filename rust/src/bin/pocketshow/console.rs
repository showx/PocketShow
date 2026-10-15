use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use eframe::egui;

#[derive(Clone, Debug)]
pub struct PersonSnap {
    pub id: i64,
    pub name: String,
    pub locked: bool,
    pub face: bool,
}

#[derive(Clone, Debug)]
pub struct CardSnap {
    pub id: String,
    pub name: String,
    pub templates: usize,
}

#[derive(Clone, Debug)]
pub struct CameraSnap {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct View {
    pub title: String,
    pub source: String,
    pub camera_id: String,
    pub cameras: Vec<CameraSnap>,
    pub paused: bool,
    pub width: u32,
    pub height: u32,
    pub yaw: f64,
    pub pitch: f64,
    pub people: Vec<PersonSnap>,
    pub gallery: Vec<CardSnap>,
    pub away: Vec<AwaySnap>,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct AwaySnap {
    pub name: String,
    pub detail: String,
    pub alarm: bool,
}

pub enum Order {
    LockAt(f32, f32),
    LockId(i64),
    Cycle(i32),
    Clear,
    Pause,
    Recenter,
    Rename(String, String),
    Delete(String),
    Camera(String),
    Pin { track: i64, person_id: String, name: String },
}

#[derive(Clone)]
pub struct Hub {
    inner: Arc<Inner>,
}

struct Inner {
    frame: Mutex<Option<(u32, u32, Vec<u8>, u64)>>,
    generation: AtomicU64,
    view: Mutex<View>,
    orders: Mutex<Vec<Order>>,
    stick: Mutex<Option<(f64, f64)>>,
    quit: AtomicBool,
    done: AtomicBool,
}

impl Hub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                frame: Mutex::new(None),
                generation: AtomicU64::new(0),
                view: Mutex::new(View {
                    title: "正在打开画面".into(),
                    source: String::new(),
                    camera_id: String::new(),
                    cameras: Vec::new(),
                    paused: false,
                    width: 0,
                    height: 0,
                    yaw: 0.0,
                    pitch: 0.0,
                    people: Vec::new(),
                    gallery: Vec::new(),
                    away: Vec::new(),
                    message: "正在加载模型…".into(),
                }),
                orders: Mutex::new(Vec::new()),
                stick: Mutex::new(None),
                quit: AtomicBool::new(false),
                done: AtomicBool::new(false),
            }),
        }
    }

    pub fn publish(&self, pixels: &[u32], width: u32, height: u32, mut view: View) {
        if view.message.is_empty() {
            view.message.clear();
        }
        let generation = self.inner.generation.fetch_add(1, Ordering::Relaxed) + 1;
        let mut rgb = Vec::with_capacity(pixels.len() * 3);
        for pixel in pixels {
            rgb.push((pixel >> 16) as u8);
            rgb.push((pixel >> 8) as u8);
            rgb.push(*pixel as u8);
        }
        *self.inner.frame.lock().expect("画面") = Some((width, height, rgb, generation));
        *self.inner.view.lock().expect("状态") = view;
    }

    pub fn note(&self, message: impl Into<String>) {
        self.inner.view.lock().expect("状态").message = message.into();
    }

    pub fn show_cameras(&self, cameras: Vec<CameraSnap>, active: String, source: String) {
        let mut view = self.inner.view.lock().expect("状态");
        view.cameras = cameras;
        view.camera_id = active;
        view.source = source;
    }

    pub fn fail(&self, message: impl Into<String>) {
        self.note(message);
        self.finish();
    }

    pub fn finish(&self) {
        self.inner.done.store(true, Ordering::Relaxed);
    }

    pub fn done(&self) -> bool {
        self.inner.done.load(Ordering::Relaxed)
    }

    pub fn request_quit(&self) {
        self.inner.quit.store(true, Ordering::Relaxed);
    }

    pub fn quit(&self) -> bool {
        self.inner.quit.load(Ordering::Relaxed)
    }

    pub fn push(&self, order: Order) {
        self.inner.orders.lock().expect("指令").push(order);
    }

    pub fn drain(&self) -> Vec<Order> {
        std::mem::take(&mut *self.inner.orders.lock().expect("指令"))
    }

    pub fn set_stick(&self, stick: Option<(f64, f64)>) {
        *self.inner.stick.lock().expect("摇杆") = stick;
    }

    pub fn stick(&self) -> Option<(f64, f64)> {
        *self.inner.stick.lock().expect("摇杆")
    }

    pub fn desktop(self) -> Result<()> {
        let native = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1180.0, 740.0])
                .with_title("PocketShow"),
            ..Default::default()
        };
        eframe::run_native(
            "PocketShow",
            native,
            Box::new(move |cc| {
                install_cjk(&cc.egui_ctx);
                Ok(Box::new(Desk {
                    hub: self,
                    texture: None,
                    seen: 0,
                    edits: HashMap::new(),
                    marking: None,
                    mark_name: String::new(),
                    tab: Tab::Live,
                }))
            }),
        )
        .map_err(|err| anyhow!("{err}"))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Live,
    People,
    Away,
    Lens,
}

struct Desk {
    hub: Hub,
    texture: Option<egui::TextureHandle>,
    seen: u64,
    edits: HashMap<String, String>,
    marking: Option<i64>,
    mark_name: String,
    tab: Tab,
}

impl eframe::App for Desk {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(33));
        if self.hub.done() && self.hub.quit() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if ctx.input(|input| input.viewport().close_requested()) {
            self.hub.request_quit();
        }
        self.keys(ctx);
        let view = self.hub.inner.view.lock().expect("状态").clone();
        let frame = self.hub.inner.frame.lock().expect("画面").clone();
        if let Some((width, height, rgb, generation)) = frame {
            if generation != self.seen && rgb.len() == (width * height * 3) as usize {
                let image = egui::ColorImage::from_rgb([width as usize, height as usize], &rgb);
                match &mut self.texture {
                    Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                    None => {
                        self.texture = Some(ctx.load_texture("view", image, egui::TextureOptions::LINEAR));
                    }
                }
                self.seen = generation;
            }
        }

        egui::SidePanel::right("side")
            .exact_width(340.0)
            .show(ctx, |ui| self.side(ui, &view));
        egui::CentralPanel::default().show(ctx, |ui| self.stage(ui, &view));
    }
}

impl Desk {
    fn keys(&mut self, ctx: &egui::Context) {
        let mut yaw = 0.0;
        let mut pitch = 0.0;
        ctx.input(|input| {
            if input.key_pressed(egui::Key::Q) || input.key_pressed(egui::Key::Escape) {
                self.hub.request_quit();
                self.hub.finish();
            }
            if input.key_pressed(egui::Key::C) {
                self.hub.push(Order::Clear);
            }
            if input.key_pressed(egui::Key::Space) {
                self.hub.push(Order::Pause);
            }
            if input.key_pressed(egui::Key::R) {
                self.hub.push(Order::Recenter);
            }
            if input.key_pressed(egui::Key::N) {
                self.hub.push(Order::Cycle(1));
            }
            if input.key_pressed(egui::Key::P) {
                self.hub.push(Order::Cycle(-1));
            }
            if input.key_down(egui::Key::ArrowLeft) {
                yaw -= 0.45;
            }
            if input.key_down(egui::Key::ArrowRight) {
                yaw += 0.45;
            }
            if input.key_down(egui::Key::ArrowUp) {
                pitch += 0.45;
            }
            if input.key_down(egui::Key::ArrowDown) {
                pitch -= 0.45;
            }
        });
        self.hub.set_stick(if yaw == 0.0 && pitch == 0.0 { None } else { Some((yaw, pitch)) });
    }

    fn stage(&mut self, ui: &mut egui::Ui, view: &View) {
        let Some(texture) = self.texture.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label(if view.message.is_empty() { "正在打开画面…" } else { view.message.as_str() });
            });
            return;
        };
        let available = ui.available_size();
        let aspect = (view.width.max(1) as f32) / (view.height.max(1) as f32);
        let mut size = available;
        if size.x / size.y > aspect {
            size.x = size.y * aspect;
        } else {
            size.y = size.x / aspect;
        }
        let response = ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click()));
        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                let rect = response.rect;
                let x = (pointer.x - rect.left()) / rect.width() * view.width as f32;
                let y = (pointer.y - rect.top()) / rect.height() * view.height as f32;
                self.hub.push(Order::LockAt(x, y));
            }
        }
    }

    fn side(&mut self, ui: &mut egui::Ui, view: &View) {
        ui.heading("PocketShow");
        ui.label(if view.source.is_empty() { "正在打开画面" } else { view.source.as_str() });
        if !view.message.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(212, 179, 122), &view.message);
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Live, "画面");
            ui.selectable_value(&mut self.tab, Tab::People, "人物");
            let alarms = view.away.iter().filter(|row| row.alarm).count();
            let away = if alarms > 0 { format!("离岗 {alarms}") } else { "离岗".into() };
            ui.selectable_value(&mut self.tab, Tab::Away, away);
            ui.selectable_value(&mut self.tab, Tab::Lens, "镜头");
        });
        ui.separator();
        egui::ScrollArea::vertical().id_salt("side-tab").show(ui, |ui| match self.tab {
            Tab::Live => self.live_tab(ui, view),
            Tab::People => self.people_tab(ui, view),
            Tab::Away => self.away_tab(ui, view),
            Tab::Lens => self.lens_tab(ui, view),
        });
    }

    fn live_tab(&mut self, ui: &mut egui::Ui, view: &View) {
        if !view.cameras.is_empty() {
            ui.label("摄像机");
            for camera in &view.cameras {
                let label = if camera.id == view.camera_id {
                    format!("当前  {}", camera.name)
                } else {
                    camera.name.clone()
                };
                if ui.button(label).clicked() {
                    self.hub.push(Order::Camera(camera.id.clone()));
                }
            }
            ui.separator();
        }
        ui.label("画面里的人");
        if view.people.is_empty() {
            ui.weak("还没有人");
        }
        for person in &view.people {
            let mark = if person.locked { "锁定" } else { "未锁" };
            let face = if person.face { "有脸" } else { "没看到脸" };
            ui.push_id(("live", person.id), |ui| {
                ui.horizontal(|ui| {
                    if ui.button(format!("{mark}  {}  {face}", person.name)).clicked() {
                        self.hub.push(Order::LockId(person.id));
                    }
                    if ui.button("标记").clicked() {
                        self.marking = Some(person.id);
                        self.mark_name.clear();
                        self.tab = Tab::People;
                    }
                });
            });
        }
        ui.horizontal(|ui| {
            if ui.button("上一个").clicked() {
                self.hub.push(Order::Cycle(-1));
            }
            if ui.button("下一个").clicked() {
                self.hub.push(Order::Cycle(1));
            }
            if ui.button("取消锁定").clicked() {
                self.hub.push(Order::Clear);
            }
        });
        if ui.button(if view.paused { "继续跟" } else { "停转" }).clicked() {
            self.hub.push(Order::Pause);
        }
    }

    fn people_tab(&mut self, ui: &mut egui::Ui, view: &View) {
        if let Some(track) = self.marking {
            if view.people.iter().any(|person| person.id == track) {
                let who = view.people.iter().find(|person| person.id == track).map(|person| person.name.as_str()).unwrap_or("这个人");
                ui.label(format!("把 {who} 记到人物库，并钉在当前画面的位置"));
                ui.text_edit_singleline(&mut self.mark_name);
                if ui.button("用这个姓名新建").clicked() {
                    self.hub.push(Order::Pin {
                        track,
                        person_id: String::new(),
                        name: self.mark_name.clone(),
                    });
                    self.marking = None;
                }
                ui.label("或记成已有的人");
                for card in &view.gallery {
                    if ui.button(format!("记成 {}", card.name)).clicked() {
                        self.hub.push(Order::Pin {
                            track,
                            person_id: card.id.clone(),
                            name: card.name.clone(),
                        });
                        self.marking = None;
                    }
                }
                if ui.button("取消标记").clicked() {
                    self.marking = None;
                }
                ui.separator();
            } else {
                self.marking = None;
            }
        } else {
            ui.weak("在「画面」里点某人的标记，再回到这里写姓名。");
            ui.separator();
        }
        ui.label("人物库");
        if view.gallery.is_empty() {
            ui.weak("还没有登记的人");
        }
        for card in &view.gallery {
            ui.push_id(&card.id, |ui| {
                let edit = self.edits.entry(card.id.clone()).or_insert_with(|| card.name.clone());
                ui.label(format!("{} · {} 条样本", card.name, card.templates));
                ui.text_edit_singleline(edit);
                ui.horizontal(|ui| {
                    if ui.button("改名").clicked() {
                        self.hub.push(Order::Rename(card.id.clone(), edit.clone()));
                    }
                    if ui.button("删除").clicked() {
                        self.hub.push(Order::Delete(card.id.clone()));
                    }
                });
            });
        }
    }

    fn away_tab(&mut self, ui: &mut egui::Ui, view: &View) {
        let here = view.away.iter().filter(|row| row.detail.starts_with("在岗")).count();
        let gone = view.away.iter().filter(|row| row.alarm || row.detail.starts_with("暂离") || row.detail.starts_with("离岗")).count();
        ui.label(format!("在岗 {here}    离开 {gone}"));
        ui.weak("工作日 9:00–18:30 离开超过 30 秒才计入今日。");
        ui.separator();
        if view.away.is_empty() {
            ui.weak("还没有钉好工位的人。到「人物」里标记后，这里才会计离岗。");
            return;
        }
        for row in &view.away {
            if row.alarm {
                ui.colored_label(egui::Color32::from_rgb(220, 120, 96), format!("{}  {}", row.name, row.detail));
            } else {
                ui.label(format!("{}  {}", row.name, row.detail));
            }
        }
    }

    fn lens_tab(&mut self, ui: &mut egui::Ui, view: &View) {
        ui.label(&view.title);
        ui.label(format!("yaw {:.2}    pitch {:.2}", view.yaw, view.pitch));
        ui.weak("按住方向转，松开停止。");
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            hold_button(ui, &self.hub, "上", 0.0, 0.45);
        });
        ui.horizontal(|ui| {
            hold_button(ui, &self.hub, "左", -0.45, 0.0);
            if ui.button("回中").clicked() {
                self.hub.push(Order::Recenter);
            }
            hold_button(ui, &self.hub, "右", 0.45, 0.0);
        });
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            hold_button(ui, &self.hub, "下", 0.0, -0.45);
        });
    }
}

fn hold_button(ui: &mut egui::Ui, hub: &Hub, label: &str, yaw: f64, pitch: f64) {
    let response = ui.add(egui::Button::new(label).min_size(egui::vec2(48.0, 32.0)));
    if response.is_pointer_button_down_on() {
        hub.set_stick(Some((yaw, pitch)));
    }
}

fn install_cjk(ctx: &egui::Context) {
    let candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ];
    let Some(bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("cjk".into(), Arc::new(egui::FontData::from_owned(bytes)));
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        family.insert(0, "cjk".into());
    }
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        family.insert(0, "cjk".into());
    }
    ctx.set_fonts(fonts);
}
