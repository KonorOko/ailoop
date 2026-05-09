use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, ops::Deref};
use syn::{
    Expr, ExprLit, Ident, Lit, Meta, PathArguments, ReturnType, Token, Type, parse::Parse,
    parse_macro_input, punctuated::Punctuated,
};

struct MacroArgs {
    name: Option<String>,
    description: Option<String>,
    param_descriptions: HashMap<String, String>,
    required: Vec<String>,
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

fn ailoop_tool_definition_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-core"], "ToolDefinition")
}

fn ailoop_tool_tag_path() -> proc_macro2::TokenStream {
    resolve_item_path(&["ailoop", "ailoop-core"], "ToolTag")
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
        let mut required = Vec::new();
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

                            required_variables.into_iter().for_each(|x| {
                                required.push(x.to_string());
                            });
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

fn get_json_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let Some(segment) = type_path.path.segments.first() else {
                return quote! { "type": "object" };
            };
            let type_name = segment.ident.to_string();

            // Handle Vec types
            if type_name == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments
                    && let Some(syn::GenericArgument::Type(inner_type)) = args.args.first()
                {
                    let inner_json_type = get_json_type(inner_type);
                    return quote! {
                        "type": "array",
                        "items": { #inner_json_type }
                    };
                }
                return quote! { "type": "array" };
            }

            // Handle primitive types
            match type_name.as_str() {
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" => {
                    quote! { "type": "number" }
                }
                "String" | "str" => {
                    quote! { "type": "string" }
                }
                "bool" => {
                    quote! { "type": "boolean" }
                }
                // Handle other types as objects
                _ => {
                    quote! { "type": "object" }
                }
            }
        }
        _ => {
            quote! { "type": "object" }
        }
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

    let required_args = args.required;

    for arg in input_fn.sig.inputs.iter_mut() {
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

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    let inner_call = if is_async {
        quote! { #fn_name(#(args.#param_names,)*).await }
    } else {
        quote! { #fn_name(#(args.#param_names,)*) }
    };

    let call_body = if is_fallible {
        quote! { #inner_call }
    } else {
        quote! { ::core::result::Result::<_, ::core::convert::Infallible>::Ok(#inner_call) }
    };

    let tool_path = ailoop_tool_path();
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
                let input_schema = serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(
                            stringify!(#param_names): {
                                #json_types,
                                "description": #param_descriptions
                            }
                        ),*
                    },
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
            ) -> impl ::core::future::Future<Output = ::core::result::Result<Self::Output, Self::Error>> + Send {
                async move { #call_body }
            }
        }

        #vis static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}
