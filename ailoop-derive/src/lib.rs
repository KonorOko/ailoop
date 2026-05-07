// use std::collections::HashMap;

// use proc_macro::TokenStream;
// use syn::{parse::Parse, parse_macro_input};

// struct MacroArgs {
//     name: Option<String>,
//     description: Option<String>,
//     param_descriptions: HashMap<String, String>,
//     required: Vec<String>,
// }

// impl Parse for MacroArgs {
//     fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {}
// }

// #[proc_macro_attribute]
// pub fn ailoop_tool(args: TokenStream, input: TokenStream) -> TokenStream {
//     let args = parse_macro_input!(args as MacroArgs);
// }
