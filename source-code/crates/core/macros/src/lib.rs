extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Object)]
pub fn derive_object(input: TokenStream) -> TokenStream {
    "fn answer() -> u32 { 42 }".parse().unwrap()
}

#[proc_macro_derive(Object3D)]
pub fn derive_object_3d(input: TokenStream) -> TokenStream {
    "fn answer() -> u32 { 42 }".parse().unwrap()
}
