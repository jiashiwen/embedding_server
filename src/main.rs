use logger::{init_log, tracing_init};
mod cmd;
mod commons;
mod configure;
mod embedding;
mod httpserver;
mod logger;
mod resources;

fn main() {
    // init_log();
    tracing_init();
    cmd::run_app();
}
