//! Per-type JSON Schema fragments for tool parameters.
//!
//! `#[ailoop_tool]` recognises the most common shapes (primitives, `Vec`,
//! `Option`, `HashMap<String, V>`, `BTreeMap<String, V>`, tuples) directly
//! from the function signature and emits the corresponding JSON Schema
//! literal at macro time. Anything else — most importantly a user-defined
//! C-style enum — is opaque to the macro, so it falls back to
//! `<T as ToolJsonType>::json_type()` and lets the type itself supply the
//! fragment. Use `#[derive(ToolJsonType)]` on enums; implement manually
//! when a custom shape is needed.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

pub trait ToolJsonType {
    /// Return a JSON Schema fragment describing this type.
    ///
    /// The returned value is expected to be a JSON object, e.g.
    /// `{"type": "string"}`. The proc macro merges a `description` field
    /// into the returned object when assembling the parameter schema.
    fn json_type() -> Value;
}

macro_rules! impl_primitive {
    ($($ty:ty),* $(,)? => $body:tt) => {
        $(
            impl ToolJsonType for $ty {
                fn json_type() -> Value {
                    json!($body)
                }
            }
        )*
    };
}

impl_primitive!(
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
    f32, f64,
    => {"type": "number"}
);

impl_primitive!(String, => {"type": "string"});
impl_primitive!(bool, => {"type": "boolean"});

impl<T: ToolJsonType> ToolJsonType for Vec<T> {
    fn json_type() -> Value {
        json!({"type": "array", "items": T::json_type()})
    }
}

impl<T: ToolJsonType> ToolJsonType for Option<T> {
    fn json_type() -> Value {
        T::json_type()
    }
}

impl<T: ToolJsonType + ?Sized> ToolJsonType for Box<T> {
    fn json_type() -> Value {
        T::json_type()
    }
}

impl<V: ToolJsonType> ToolJsonType for HashMap<String, V> {
    fn json_type() -> Value {
        json!({"type": "object", "additionalProperties": V::json_type()})
    }
}

impl<V: ToolJsonType> ToolJsonType for BTreeMap<String, V> {
    fn json_type() -> Value {
        json!({"type": "object", "additionalProperties": V::json_type()})
    }
}

macro_rules! impl_tuple {
    ($($ty:ident),+ $(,)?) => {
        impl<$($ty: ToolJsonType),+> ToolJsonType for ($($ty,)+) {
            fn json_type() -> Value {
                // JSON Schema 2020-12: positional item types live under
                // `prefixItems`; the legacy `items: [...]` array form was
                // dropped. Both Anthropic and OpenAI tool/function-calling
                // accept arbitrary JSON Schema, so we emit the modern form
                // and rely on the model — not a strict validator — to read
                // it.
                json!({
                    "type": "array",
                    "prefixItems": [$($ty::json_type()),+]
                })
            }
        }
    };
}

impl_tuple!(A);
impl_tuple!(A, B);
impl_tuple!(A, B, C);
impl_tuple!(A, B, C, D);
impl_tuple!(A, B, C, D, E);
impl_tuple!(A, B, C, D, E, F);
impl_tuple!(A, B, C, D, E, F, G);
impl_tuple!(A, B, C, D, E, F, G, H);
