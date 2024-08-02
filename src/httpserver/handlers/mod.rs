mod config;
mod handler_embedding;
mod handler_root;

use crate::httpserver::module::Response;
use axum::Json;
pub use config::current_config;
pub use handler_embedding::*;
pub use handler_root::root;

type HandlerResult<T> = crate::httpserver::module::Result<Json<Response<T>>>;
