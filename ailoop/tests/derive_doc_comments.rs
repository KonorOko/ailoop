use ailoop::{Tool, ailoop_tool};

/// Multiplies two numbers together.
#[ailoop_tool]
async fn multiply(
    #[doc = "The first factor."] a: i32,
    #[doc = "The second factor."] b: i32,
) -> i32 {
    a * b
}

/// Doc comment ignored when description= wins.
#[ailoop_tool(description = "explicit wins")]
async fn explicit_description(x: i32) -> i32 {
    x
}

#[ailoop_tool]
async fn no_doc_no_attr(x: i32) -> i32 {
    x
}

#[ailoop_tool(params(a = "explicit param"))]
async fn param_explicit_wins(#[doc = "ignored doc-comment"] a: i32) -> i32 {
    a
}

/// Multi-line description
/// that spans two lines.
#[ailoop_tool]
async fn multiline() -> i32 {
    1
}

#[test]
fn doc_comment_falls_back_for_fn_description() {
    let def = Multiply.definition();
    assert_eq!(def.description, "Multiplies two numbers together.");
}

#[test]
fn explicit_description_wins_over_doc_comment() {
    let def = ExplicitDescription.definition();
    assert_eq!(def.description, "explicit wins");
}

#[test]
fn no_doc_no_description_yields_empty_string() {
    let def = NoDocNoAttr.definition();
    assert_eq!(def.description, "");
}

#[test]
fn doc_comment_falls_back_for_param_descriptions() {
    let def = Multiply.definition();
    assert_eq!(
        def.input_schema["properties"]["a"]["description"],
        "The first factor."
    );
    assert_eq!(
        def.input_schema["properties"]["b"]["description"],
        "The second factor."
    );
}

#[test]
fn explicit_param_description_wins_over_doc_comment() {
    let def = ParamExplicitWins.definition();
    assert_eq!(
        def.input_schema["properties"]["a"]["description"],
        "explicit param"
    );
}

#[test]
fn no_doc_no_attr_param_yields_empty_string() {
    let def = NoDocNoAttr.definition();
    assert_eq!(def.input_schema["properties"]["x"]["description"], "");
}

#[test]
fn multi_line_doc_comment_preserved_with_newline() {
    let def = Multiline.definition();
    assert_eq!(
        def.description,
        "Multi-line description\nthat spans two lines."
    );
}
