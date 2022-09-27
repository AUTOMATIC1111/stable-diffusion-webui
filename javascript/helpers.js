// helper functions

function debounce(func, wait_time) {
	let timeout;
	return function wrapped(...args) {
		let call_function = () => {
			clearTimeout(timeout);
			func(...args)
		}
		clearTimeout(timeout);
		timeout = setTimeout(call_function, wait_time);
	};
}