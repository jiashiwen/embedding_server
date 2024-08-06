mod common_module;
pub mod module_retriever;
mod module_task;
mod request_module;
mod response_module;

pub use request_module::*;
pub use response_module::*;

/// 定义自己的 Result
pub type Result<T> = std::result::Result<T, crate::httpserver::exception::AppError>;
