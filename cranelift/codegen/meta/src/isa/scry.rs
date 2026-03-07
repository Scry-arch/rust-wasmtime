use crate::cdsl::isa::TargetIsa;
use crate::cdsl::settings::SettingGroupBuilder;

pub(crate) fn define() -> TargetIsa {
    let setting = SettingGroupBuilder::new("scry");
    
    TargetIsa::new("scry", setting.build())
}
