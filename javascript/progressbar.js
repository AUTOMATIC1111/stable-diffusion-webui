// code related to showing and updating progressbar shown as the image is being made

(function() {
var global_progressbars = {};

var avl_progressbars = "txt2img|img2img|ti".split("|");
let title;

onLoad(function() {
    title = document.title;
    avl_progressbars.forEach((id) => {
        var bar = gradioApp().getElementById(id+"_progressbar");
        var skip = gradioApp().getElementById(id+"_skip");
        var interrupt = gradioApp().getElementById(id+"_interrupt");

        var mo = new MutationObserver(function(){
            var is_progress = !!gradioApp().getElementById(id+"_progress_span");
            if (skip)skip.style.display = is_progress?"block":"none";
            interrupt.style.display = is_progress?"block":"none";

            var preview = gradioApp().getElementById(id+"_preview");
            var gallery = gradioApp().getElementById(id+"_gallery");

            if(preview != null && gallery != null){
                preview.style.width = gallery.clientWidth + "px";
                preview.style.height = gallery.clientHeight + "px";
            }

            setTimeout(function progressLoop() {
                var btn = gradioApp().getElementById(id+"_check_progress");
                if(btn==null) return;
                btn.click();
            }, 500);
        });

        mo.observe(bar, {childList:true});
        global_progressbars[id]=bar;
    });
});

function check_progressbar(prefix) {
    var bar = global_progressbars[prefix];

    if(opts.show_progress_in_title && bar && bar.offsetParent) {
        if(bar.innerText) document.title =title + ' - ' + bar.innerText;
        else document.title = title;
    }
}

onUiUpdate(function(){
    check_progressbar('txt2img');
    check_progressbar('img2img');
    check_progressbar('ti');
})

})();

function requestProgress(id) {
    var btn = gradioApp().getElementById(id+"_check_progress_initial");
    btn&&btn.click();
}