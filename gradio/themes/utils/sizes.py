from __future__ import annotations


class Size:
    all = []

    def __init__(
        self, xxs: str, xs: str, sm: str, md: str, lg: str, xl: str, xxl: str, name=None
    ):
        self.xxs = xxs
        self.xs = xs
        self.sm = sm
        self.md = md
        self.lg = lg
        self.xl = xl
        self.xxl = xxl
        self.name = name
        Size.all.append(self)

    def expand(self) -> list[str]:
        return [self.xxs, self.xs, self.sm, self.md, self.lg, self.xl, self.xxl]


radius_none = Size(
    name="radius_none",
    xxs="0px",
    xs="0px",
    sm="0px",
    md="0px",
    lg="0px",
    xl="0px",
    xxl="0px",
)

radius_sm = Size(
    name="radius_sm",
    xxs="1px",
    xs="1px",
    sm="2px",
    md="4px",
    lg="6px",
    xl="8px",
    xxl="12px",
)

radius_md = Size(
    name="radius_md",
    xxs="1px",
    xs="2px",
    sm="4px",
    md="6px",
    lg="8px",
    xl="12px",
    xxl="22px",
)

radius_lg = Size(
    name="radius_lg",
    xxs="2px",
    xs="4px",
    sm="6px",
    md="8px",
    lg="12px",
    xl="16px",
    xxl="24px",
)

spacing_sm = Size(
    name="spacing_sm",
    xxs="1px",
    xs="1px",
    sm="2px",
    md="4px",
    lg="6px",
    xl="9px",
    xxl="12px",
)

spacing_md = Size(
    name="spacing_md",
    xxs="1px",
    xs="2px",
    sm="4px",
    md="6px",
    lg="8px",
    xl="10px",
    xxl="16px",
)

spacing_lg = Size(
    name="spacing_lg",
    xxs="2px",
    xs="4px",
    sm="6px",
    md="8px",
    lg="10px",
    xl="14px",
    xxl="28px",
)

text_sm = Size(
    name="text_sm",
    xxs="8px",
    xs="9px",
    sm="11px",
    md="13px",
    lg="16px",
    xl="20px",
    xxl="24px",
)

text_md = Size(
    name="text_md",
    xxs="9px",
    xs="10px",
    sm="12px",
    md="14px",
    lg="16px",
    xl="22px",
    xxl="26px",
)

text_lg = Size(
    name="text_lg",
    xxs="10px",
    xs="12px",
    sm="14px",
    md="16px",
    lg="20px",
    xl="24px",
    xxl="28px",
)
