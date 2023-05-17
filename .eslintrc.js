module.exports = {
    env: {
        browser: true,
        es2021: true,
    },
    // "extends": "eslint:recommended",
    parserOptions: {
        ecmaVersion: "latest",
    },
    rules: {
        "arrow-spacing": "error",
        "block-spacing": "error",
        "brace-style": "error",
        "comma-dangle": ["error", "only-multiline"],
        "comma-spacing": "error",
        "comma-style": ["error", "last"],
        "curly": ["error", "multi-line", "consistent"],
        "eol-last": "error",
        "func-call-spacing": "error",
        "function-call-argument-newline": ["error", "consistent"],
        "function-paren-newline": ["error", "consistent"],
        "indent": ["error", 4],
        "key-spacing": "error",
        "keyword-spacing": "error",
        "linebreak-style": ["error", "unix"],
        "no-extra-semi": "error",
        "no-mixed-spaces-and-tabs": "error",
        "no-trailing-spaces": "error",
        "no-whitespace-before-property": "error",
        "object-curly-newline": ["error", {consistent: true, multiline: true}],
        "quote-props": ["error", "consistent-as-needed"],
        "semi": ["error", "always"],
        "semi-spacing": "error",
        "semi-style": ["error", "last"],
        "space-before-blocks": "error",
        "space-before-function-paren": ["error", "never"],
        "space-in-parens": ["error", "never"],
        "space-infix-ops": "error",
        "space-unary-ops": "error",
        "switch-colon-spacing": "error",
        "template-curly-spacing": ["error", "never"],
        "unicode-bom": "error",
        // "no-multi-spaces": "error", // TODO: enable?
        // "object-curly-spacing": "off", // TODO: enable?
        // "object-property-newline": "off", // TODO: enable?
        // "operator-linebreak": "off", // TODO: enable?
        // "quotes": ["error", "double", {avoidEscape: true}],  // TODO: enable?
    },
};
