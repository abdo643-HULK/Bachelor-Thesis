pub mod primitives;

trait Renderer {}

pub struct Renderer3D {}

impl Renderer for Renderer3D {}

#[derivative(Default)]
#[derive(Debug, Clone, Hash)]
pub struct WgpuSettings {
    pub device_label: Option<Cow<'static, str>>,
    pub backends: Option<Backends>,
    pub power_preference: PowerPreference,
    pub priority: WgpuSettingsPriority,
    /// The features to ensure are enabled regardless of what the adapter/backend supports.
    /// Setting these explicitly may cause renderer initialization to fail.
    pub features: WgpuFeatures,
    /// The features to ensure are disabled regardless of what the adapter/backend supports
    pub disabled_features: Option<WgpuFeatures>,
    /// The imposed limits.
    pub limits: WgpuLimits,
    /// The constraints on limits allowed regardless of what the adapter/backend supports
    pub constrained_limits: Option<WgpuLimits>,
}
