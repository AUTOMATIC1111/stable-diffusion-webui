function t(n,r){if(n==null)return null;if(typeof n=="string")return{name:"file_data",data:n};if(n.is_file)n.data=r+"file="+n.name;else if(Array.isArray(n))for(const a of n)t(a,r);return n}export{t as n};
//# sourceMappingURL=utils.27234e1d.js.map
