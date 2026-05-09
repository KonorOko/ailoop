use std::collections::{BTreeMap, HashMap};

use ailoop::{Tool, ToolJsonType, ailoop_tool};
use serde::Deserialize;
use serde_json::json;

#[ailoop_tool]
async fn maybe_greet(name: String, greeting: Option<String>) -> i32 {
    let _ = (name, greeting);
    0
}

#[ailoop_tool]
async fn lookup(values: HashMap<String, i64>) -> i32 {
    let _ = values;
    0
}

#[ailoop_tool]
async fn lookup_btree(values: BTreeMap<String, String>) -> i32 {
    let _ = values;
    0
}

#[ailoop_tool]
async fn pair(point: (i32, String)) -> i32 {
    let _ = point;
    0
}

#[derive(Debug, Deserialize, ToolJsonType)]
enum Color {
    Red,
    Green,
    Blue,
}

#[ailoop_tool]
async fn paint(color: Color) -> i32 {
    let _ = color;
    0
}

#[ailoop_tool(required(name))]
async fn explicit_required(name: String, greeting: Option<String>) -> i32 {
    let _ = (name, greeting);
    0
}

#[ailoop_tool(required(name, greeting))]
async fn explicit_required_overrides_inference(name: String, greeting: Option<String>) -> i32 {
    let _ = (name, greeting);
    0
}

#[ailoop_tool(required())]
async fn explicit_empty_required(name: String) -> i32 {
    let _ = name;
    0
}

#[test]
fn option_param_is_not_required_but_other_params_are() {
    let def = MaybeGreet.definition();
    let required = def.input_schema["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["name"]);
}

#[test]
fn option_param_schema_uses_inner_type() {
    let def = MaybeGreet.definition();
    assert_eq!(
        def.input_schema["properties"]["greeting"]["type"], "string",
        "Option<String> should describe the inner type, not 'object'"
    );
}

#[test]
fn hashmap_string_v_renders_as_object_with_additional_properties() {
    let def = Lookup.definition();
    let schema = &def.input_schema["properties"]["values"];
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], json!({"type": "number"}));
}

#[test]
fn btreemap_string_v_renders_as_object_with_additional_properties() {
    let def = LookupBtree.definition();
    let schema = &def.input_schema["properties"]["values"];
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], json!({"type": "string"}));
}

#[test]
fn tuple_param_renders_as_array_with_prefix_items() {
    let def = Pair.definition();
    let schema = &def.input_schema["properties"]["point"];
    assert_eq!(schema["type"], "array");
    assert_eq!(
        schema["prefixItems"],
        json!([{"type": "number"}, {"type": "string"}])
    );
}

#[test]
fn c_style_enum_renders_as_string_enum_via_derive() {
    let def = Paint.definition();
    let schema = &def.input_schema["properties"]["color"];
    assert_eq!(schema["type"], "string");
    assert_eq!(schema["enum"], json!(["Red", "Green", "Blue"]));
}

#[test]
fn explicit_required_subset_wins_over_inference() {
    // Inference would mark `name` required; the explicit list does the
    // same here, so the override should match.
    let def = ExplicitRequired.definition();
    let required = def.input_schema["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["name"]);
}

#[test]
fn explicit_required_can_force_optional_to_required() {
    // The user can override inference to require an `Option<T>` parameter.
    let def = ExplicitRequiredOverridesInference.definition();
    let required = def.input_schema["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["name", "greeting"]);
}

#[test]
fn explicit_empty_required_overrides_inference() {
    // Even when inference would mark `name` as required, an explicit empty
    // `required()` wins and the array stays empty.
    let def = ExplicitEmptyRequired.definition();
    assert!(def.input_schema["required"].as_array().unwrap().is_empty());
}

#[test]
fn tojsontype_blanket_impls_compose() {
    // The trait impls in ailoop-tools should compose for nested generic
    // shapes, which is what a manual `impl ToolJsonType` would lean on.
    assert_eq!(
        <Vec<Color>>::json_type(),
        json!({
            "type": "array",
            "items": {"type": "string", "enum": ["Red", "Green", "Blue"]}
        })
    );
    assert_eq!(
        <Option<HashMap<String, i64>>>::json_type(),
        json!({"type": "object", "additionalProperties": {"type": "number"}})
    );
}
