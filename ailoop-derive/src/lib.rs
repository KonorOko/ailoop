use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, ops::Deref};
use syn::{
    Expr, ExprLit, GenericArgument, Ident, Lit, Meta, PathArguments, ReturnType, Token, Type,
    parse::Parse, parse_macro_input, punctuated::Punctuated,
};

struct MacroArgs {
    name: Option<String>,
    description: Option<String>,
    param_descriptions: HashMap<String, String>,
    /// `None` when the user didn't write `required(...)` — the macro
    /// infers from parameter types. `Some(_)` (even if empty) means the
    /// caller is taking explicit control.
    required: Option<Vec<String>>,
    tags: Vec<Ident>,
}

/// Resolve the path to an item that lives in (or is re-exported by) one of the
/// given crates. The crates are tried in order; the first one available in the
/// caller's `Cargo.toml` wins. The last entry is also used as the literal
/// fallback if nothing is found.
fn resolve_item_path(candidate_crates: &[&str], item: &str) -> proc_macro2::TokenStream {
    let item_ident = format_ident!("{item}");

    for c in candidate_crates {
        if let Ok(found) = proc_macro_crate::crate_name(c) {
            return match found {
                proc_macro_crate::FoundCrate::Itself => quote!(crate::#item_ident),
                proc_macro_crate::FoundCrate::Name(name) => {
                    let crate_ident = format_ident!("{name}");
                    quote!(::#crate_ident::#item_ident)
                }
            };
        }
    }

    let last = candidate_crates
        .last()
        .expect("resolve_item_path needs at least one candidate crate");
    let crate_ident = format_ident!("{}", last.replace('-', "_"));
    quote!(::#crate_ident::#item_ident)
}

fn ailoop_tool_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-tools"], "Tool")
}

fn ailoop_tool_context_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-tools"], "ToolContext")
}

fn ailoop_tool_definition_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-core"], "ToolDefinition")
}

fn ailoop_tool_tag_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-core"], "ToolTag")
}

fn ailoop_tool_json_type_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-tools"], "ToolJsonType")
}

/// Returns `true` if `ty` is a reference whose final path segment is
/// `ToolContext` — this is what the macro looks for in the user's fn
/// signature to detect an opt-in `ctx: &ToolContext` trailing parameter.
/// Any path prefix matches (`ToolContext`, `ailoop_tools::ToolContext`,
/// `ailoop::ToolContext`) so users don't have to import the type a
/// specific way.
fn is_tool_context_ref(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    let Type::Path(type_path) = &*reference.elem else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .map(|seg| seg.ident == "ToolContext")
        .unwrap_or(false)
}

fn extract_doc_comment(attrs: &[syn::Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();

    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) = &nv.value
        else {
            continue;
        };
        let raw = s.value();
        // `///` desugars to `#[doc = " text"]`; drop that conventional single
        // leading space so the rendered description matches the source.
        let line = raw.strip_prefix(' ').unwrap_or(&raw).to_string();
        lines.push(line);
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn parse_string_literal(expr: &Expr, field_name: &str) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit_str),
            ..
        }) => Ok(lit_str.value()),
        _ => Err(syn::Error::new_spanned(
            expr,
            format!("`{field_name}` must be a string literal"),
        )),
    }
}

fn validate_explicit_tool_name(name: &str, expr: &Expr) -> syn::Result<()> {
    if name.is_empty() || name.len() > 64 {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    }

    let mut chars = name.chars();
    let Some(first_char) = chars.next() else {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    };

    if !first_char.is_ascii_alphabetic() && first_char != '_' {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must start with an ASCII letter or underscore",
        ));
    }

    if chars.any(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-') {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` may only contain ASCII letters, digits, underscores, or hyphens",
        ));
    }

    Ok(())
}

impl Parse for MacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut param_descriptions = HashMap::new();
        let mut required: Option<Vec<String>> = None;
        let mut tags = Vec::new();

        // If the input is empty, return default values
        if input.is_empty() {
            return Ok(MacroArgs {
                name,
                description,
                param_descriptions,
                required,
                tags,
            });
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &nv.path,
                            "unsupported top-level #[ailoop_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "name" => {
                            let parsed_name = parse_string_literal(&nv.value, "name")?;
                            validate_explicit_tool_name(&parsed_name, &nv.value)?;
                            name = Some(parsed_name);
                        }
                        "description" => {
                            description = Some(parse_string_literal(&nv.value, "description")?);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &nv.path,
                                format!(
                                    "unsupported top-level #[ailoop_tool] argument `{}`",
                                    ident
                                ),
                            ));
                        }
                    }
                }
                Meta::List(list) => {
                    let ident = list.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &list.path,
                            "unsupported top-level #[ailoop_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "params" => {
                            let nested: Punctuated<Meta, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            for meta in nested {
                                if let Meta::NameValue(nv) = meta
                                    && let Expr::Lit(ExprLit {
                                        lit: Lit::Str(lit_str),
                                        ..
                                    }) = nv.value
                                {
                                    let Some(param_ident) = nv.path.get_ident() else {
                                        return Err(syn::Error::new_spanned(
                                            &nv.path,
                                            "parameter descriptions must use identifier keys",
                                        ));
                                    };
                                    let param_name = param_ident.to_string();
                                    param_descriptions.insert(param_name, lit_str.value());
                                }
                            }
                        }
                        "required" => {
                            let required_variables: Punctuated<Ident, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            let names: Vec<String> = required_variables
                                .into_iter()
                                .map(|x| x.to_string())
                                .collect();
                            required = Some(names);
                        }
                        "tags" => {
                            let tag_idents: Punctuated<Ident, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            tag_idents.into_iter().for_each(|x| {
                                tags.push(x);
                            });
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &list.path,
                                format!(
                                    "unsupported top-level #[ailoop_tool] argument `{}`",
                                    ident
                                ),
                            ));
                        }
                    }
                }
                Meta::Path(path) => {
                    let message = if let Some(ident) = path.get_ident() {
                        format!("unsupported top-level #[ailoop_tool] argument `{ident}`")
                    } else {
                        "unsupported top-level #[ailoop_tool] argument".to_string()
                    };

                    return Err(syn::Error::new_spanned(path, message));
                }
            }
        }

        Ok(MacroArgs {
            name,
            description,
            param_descriptions,
            required,
            tags,
        })
    }
}

/// Pull the inner type out of an `Option<T>` shell. Returns `Some(&T)` only
/// for path types whose final segment is `Option`; nested module paths like
/// `core::option::Option<T>` also work (we look at the last segment).
fn unwrap_option(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last = type_path.path.segments.last()?;
    if last.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

fn is_option_type(ty: &Type) -> bool {
    unwrap_option(ty).is_some()
}

fn first_generic_type(args: &PathArguments) -> Option<&Type> {
    let PathArguments::AngleBracketed(generics) = args else {
        return None;
    };
    generics.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

fn first_two_generic_types(args: &PathArguments) -> Option<(&Type, &Type)> {
    let PathArguments::AngleBracketed(generics) = args else {
        return None;
    };
    let mut types = generics.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    });
    let first = types.next()?;
    let second = types.next()?;
    Some((first, second))
}

fn is_string_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    segment.ident == "String"
}

fn fallback_via_trait(ty: &Type) -> proc_macro2::TokenStream {
    let trait_path = ailoop_tool_json_type_path();
    quote! { <#ty as #trait_path>::json_type() }
}

/// Build a `TokenStream` that evaluates to a `serde_json::Value` describing
/// the JSON Schema fragment for `ty`. Returns either a literal `json!{...}`
/// expression (for shapes the macro recognises) or a runtime
/// `<T as ToolJsonType>::json_type()` dispatch (for everything else).
fn get_json_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Tuple(tuple) => {
            // JSON Schema 2020-12 uses `prefixItems` for positional item
            // types; the legacy `items: [...]` array form is no longer
            // valid. Anthropic and OpenAI tool definitions accept arbitrary
            // JSON Schema and route it to the model rather than a strict
            // validator, so the modern form is the portable choice.
            let inner: Vec<proc_macro2::TokenStream> =
                tuple.elems.iter().map(get_json_type).collect();
            quote! {
                ::serde_json::json!({
                    "type": "array",
                    "prefixItems": [#(#inner),*]
                })
            }
        }
        Type::Path(type_path) => {
            let Some(segment) = type_path.path.segments.last() else {
                return fallback_via_trait(ty);
            };
            let type_name = segment.ident.to_string();

            match type_name.as_str() {
                "Vec" => {
                    if let Some(inner_ty) = first_generic_type(&segment.arguments) {
                        let inner = get_json_type(inner_ty);
                        return quote! {
                            ::serde_json::json!({
                                "type": "array",
                                "items": #inner
                            })
                        };
                    }
                    quote! { ::serde_json::json!({"type": "array"}) }
                }
                "Option" => {
                    // `Option<T>` only affects required-ness; the schema
                    // describes the inner type. The proc macro keeps
                    // optional params out of the `required` list separately.
                    if let Some(inner_ty) = first_generic_type(&segment.arguments) {
                        return get_json_type(inner_ty);
                    }
                    fallback_via_trait(ty)
                }
                "HashMap" | "BTreeMap" => {
                    if let Some((k, v)) = first_two_generic_types(&segment.arguments)
                        && is_string_type(k)
                    {
                        let v_schema = get_json_type(v);
                        return quote! {
                            ::serde_json::json!({
                                "type": "object",
                                "additionalProperties": #v_schema
                            })
                        };
                    }
                    fallback_via_trait(ty)
                }
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" | "f32" | "f64" => {
                    quote! { ::serde_json::json!({"type": "number"}) }
                }
                "String" | "str" => {
                    quote! { ::serde_json::json!({"type": "string"}) }
                }
                "bool" => {
                    quote! { ::serde_json::json!({"type": "boolean"}) }
                }
                _ => fallback_via_trait(ty),
            }
        }
        _ => fallback_via_trait(ty),
    }
}

struct ReturnInfo {
    output: proc_macro2::TokenStream,
    error: proc_macro2::TokenStream,
    is_fallible: bool,
}

fn analyze_return_type(return_type: &ReturnType) -> syn::Result<ReturnInfo> {
    let infallible = quote!(::core::convert::Infallible);

    let ReturnType::Type(_, ty) = return_type else {
        return Ok(ReturnInfo {
            output: quote!(()),
            error: infallible,
            is_fallible: false,
        });
    };

    if let Type::Path(type_path) = ty.deref()
        && let Some(last_segment) = type_path.path.segments.last()
        && last_segment.ident == "Result"
    {
        let PathArguments::AngleBracketed(args) = &last_segment.arguments else {
            return Err(syn::Error::new_spanned(
                &last_segment.arguments,
                "expected angle-bracketed type parameters for Result<T, E>",
            ));
        };

        let mut generic_args = args.args.iter();
        let (Some(output), Some(error)) = (generic_args.next(), generic_args.next()) else {
            return Err(syn::Error::new_spanned(
                &args.args,
                "expected Result<T, E> with exactly two type parameters",
            ));
        };

        if generic_args.next().is_some() {
            return Err(syn::Error::new_spanned(
                &args.args,
                "expected Result<T, E> with exactly two type parameters",
            ));
        }

        return Ok(ReturnInfo {
            output: quote!(#output),
            error: quote!(#error),
            is_fallible: true,
        });
    }

    Ok(ReturnInfo {
        output: quote!(#ty),
        error: infallible,
        is_fallible: false,
    })
}

#[proc_macro_attribute]
pub fn ailoop_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as MacroArgs);
    let mut input_fn = parse_macro_input!(input as syn::ItemFn);

    let fn_name = input_fn.sig.ident.clone();
    let fn_name_str = fn_name.to_string();
    let tool_name = args.name.clone().unwrap_or_else(|| fn_name_str.clone());
    let vis = input_fn.vis.clone();
    let is_async = input_fn.sig.asyncness.is_some();

    let return_type = &input_fn.sig.output;
    let return_info = match analyze_return_type(return_type) {
        Ok(info) => info,
        Err(error) => return error.into_compile_error().into(),
    };
    let output_type = return_info.output;
    let error_type = return_info.error;
    let is_fallible = return_info.is_fallible;

    let struct_name = format_ident!("{}", { fn_name_str.to_case(Case::Pascal) });

    let tool_description: String = args
        .description
        .or_else(|| extract_doc_comment(&input_fn.attrs))
        .unwrap_or_default();

    let mut param_names: Vec<Ident> = Vec::new();
    let mut param_types: Vec<Type> = Vec::new();
    let mut param_descriptions: Vec<String> = Vec::new();
    let mut json_types: Vec<proc_macro2::TokenStream> = Vec::new();

    // Detect a trailing `ctx: &ToolContext` parameter. When present,
    // the macro routes the engine-supplied `ToolContext` through to
    // the user's function and excludes it from the generated `Args`
    // struct (the model never sees this parameter). When absent, the
    // generated `Tool::call` still receives a `_ctx: &ToolContext`
    // (silently ignored) so the trait signature is uniform.
    let total_inputs = input_fn.sig.inputs.len();
    let takes_ctx = match input_fn.sig.inputs.iter().last() {
        Some(syn::FnArg::Typed(pat_type)) => is_tool_context_ref(&pat_type.ty),
        _ => false,
    };
    let payload_inputs = if takes_ctx {
        total_inputs - 1
    } else {
        total_inputs
    };

    for (idx, arg) in input_fn.sig.inputs.iter_mut().enumerate() {
        if takes_ctx && idx == payload_inputs {
            // Skip the trailing ctx param — it is not a model-visible argument.
            continue;
        }
        let syn::FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let syn::Pat::Ident(param_ident) = &*pat_type.pat else {
            continue;
        };
        let param_name = param_ident.ident.clone();
        let param_name_str = param_name.to_string();
        let ty = (*pat_type.ty).clone();
        let json_type = get_json_type(&ty);
        let description = args
            .param_descriptions
            .get(&param_name_str)
            .cloned()
            .or_else(|| extract_doc_comment(&pat_type.attrs))
            .unwrap_or_default();

        // Rust forbids `#[doc = ...]` (and `///`) attributes on fn
        // parameters at compile time, so strip them after harvesting the
        // description — they only exist for the macro to read.
        pat_type.attrs.retain(|a| !a.path().is_ident("doc"));

        param_names.push(param_name);
        param_types.push(ty);
        param_descriptions.push(description);
        json_types.push(json_type);
    }

    // Required-ness: explicit `required(...)` from the attribute wins; if
    // omitted, infer from parameter types — anything not `Option<T>` is
    // required.
    let required_args: Vec<String> = match args.required {
        Some(explicit) => explicit,
        None => param_names
            .iter()
            .zip(param_types.iter())
            .filter(|(_, ty)| !is_option_type(ty))
            .map(|(name, _)| name.to_string())
            .collect(),
    };

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    let inner_call = match (is_async, takes_ctx) {
        (true, true) => quote! { #fn_name(#(args.#param_names,)* ctx).await },
        (true, false) => quote! { #fn_name(#(args.#param_names,)*).await },
        (false, true) => quote! { #fn_name(#(args.#param_names,)* ctx) },
        (false, false) => quote! { #fn_name(#(args.#param_names,)*) },
    };

    let call_body = if is_fallible {
        quote! { #inner_call }
    } else {
        quote! { ::core::result::Result::<_, ::core::convert::Infallible>::Ok(#inner_call) }
    };

    let tool_path = ailoop_tool_path();
    let tool_context_path = ailoop_tool_context_path();
    let tool_definition_path = ailoop_tool_definition_path();
    let tool_tag_path = ailoop_tool_tag_path();

    let tags_expr = if args.tags.is_empty() {
        quote! { ::std::vec::Vec::new() }
    } else {
        let tag_paths: Vec<proc_macro2::TokenStream> = args
            .tags
            .iter()
            .map(|v| quote!(#tool_tag_path::#v))
            .collect();
        quote! { ::std::vec![#(#tag_paths),*] }
    };

    // Build the per-parameter property objects at runtime. Each `json_type`
    // expression evaluates to a `serde_json::Value`; we splice the
    // description in as an extra key. Per-parameter Values are inserted
    // into a `Map` so the final `serde_json::json!` call sees a complete,
    // ordered properties object.
    let property_inserts = param_names
        .iter()
        .zip(json_types.iter())
        .zip(param_descriptions.iter())
        .map(|((name, json_type), desc)| {
            let name_str = name.to_string();
            quote! {
                {
                    let mut __schema: ::serde_json::Value = #json_type;
                    if let ::serde_json::Value::Object(ref mut __obj) = __schema {
                        __obj.insert(
                            ::std::string::String::from("description"),
                            ::serde_json::Value::String(::std::string::String::from(#desc)),
                        );
                    }
                    __properties.insert(::std::string::String::from(#name_str), __schema);
                }
            }
        });

    let expanded = quote! {
        #[derive(serde::Deserialize)]
        #vis struct #params_struct_name {
            #(#vis #param_names: #param_types,)*
        }

        #input_fn

        #[derive(Default)]
        #vis struct #struct_name;

        impl #tool_path for #struct_name {
            const NAME: &'static str = #tool_name;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = #error_type;

            fn definition(&self) -> #tool_definition_path {
                let mut __properties: ::serde_json::Map<::std::string::String, ::serde_json::Value> =
                    ::serde_json::Map::new();
                #(#property_inserts)*

                let input_schema = ::serde_json::json!({
                    "type": "object",
                    "properties": __properties,
                    "required": [#(#required_args),*]
                });

                #tool_definition_path::new(
                    #tool_name,
                    #tool_description,
                    input_schema,
                    #tags_expr,
                )
            }

            fn call(
                &self,
                args: Self::Args,
                #[allow(unused_variables)] ctx: &#tool_context_path,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<Self::Output, Self::Error>> + Send {
                async move { #call_body }
            }
        }

        #vis static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(ToolJsonType)]
pub fn derive_tool_json_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    let trait_path = ailoop_tool_json_type_path();
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let body = match &input.data {
        syn::Data::Enum(data_enum) => {
            let mut variant_names: Vec<String> = Vec::new();
            for variant in &data_enum.variants {
                match &variant.fields {
                    syn::Fields::Unit => variant_names.push(variant.ident.to_string()),
                    syn::Fields::Named(_) | syn::Fields::Unnamed(_) => {
                        return syn::Error::new_spanned(
                            variant,
                            "#[derive(ToolJsonType)] only supports C-style enums (unit variants). \
                             Variants carrying a payload are out of scope — implement \
                             `ToolJsonType` manually for this type.",
                        )
                        .into_compile_error()
                        .into();
                    }
                }
            }
            quote! {
                ::serde_json::json!({
                    "type": "string",
                    "enum": [#(#variant_names),*]
                })
            }
        }
        syn::Data::Struct(_) | syn::Data::Union(_) => {
            return syn::Error::new_spanned(
                &input.ident,
                "#[derive(ToolJsonType)] is only supported on enums; implement `ToolJsonType` \
                 manually for structs or unions.",
            )
            .into_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
            fn json_type() -> ::serde_json::Value {
                #body
            }
        }
    };

    TokenStream::from(expanded)
}
