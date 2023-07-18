(function() {
    var ignore = localStorage.getItem("bad-scale-ignore-it") == "ignore-it";

    function getScale() {
        var ratio = 0,
            screen = window.screen,
            ua = navigator.userAgent.toLowerCase();

        if (window.devicePixelRatio !== undefined) {
            ratio = window.devicePixelRatio;
        } else if (~ua.indexOf('msie')) {
            if (screen.deviceXDPI && screen.logicalXDPI) {
                ratio = screen.deviceXDPI / screen.logicalXDPI;
            }
        } else if (window.outerWidth !== undefined && window.innerWidth !== undefined) {
            ratio = window.outerWidth / window.innerWidth;
        }

        return ratio == 0 ? 0 : Math.round(ratio * 100);
    }

    var showing = false;

    var div = document.createElement("div");
    div.style.position = "fixed";
    div.style.top = "0px";
    div.style.left = "0px";
    div.style.width = "100vw";
    div.style.backgroundColor = "firebrick";
    div.style.textAlign = "center";
    div.style.zIndex = 99;

    var b = document.createElement("b");
    b.innerHTML = 'Bad Scale: ??% ';

    div.appendChild(b);

    var note1 = document.createElement("p");
    note1.innerHTML = "Change your browser or your computer settings!";
    note1.title = 'Just make sure "computer-scale" * "browser-scale" = 100% ,\n' +
        "you can keep your computer-scale and only change this page's scale,\n" +
        "for example: your computer-scale is 125%, just use [\"CTRL\"+\"-\"] to make your browser-scale of this page to 80%.";
    div.appendChild(note1);

    var note2 = document.createElement("p");
    note2.innerHTML = " Otherwise, it will cause this page to not function properly!";
    note2.title = "When you click \"Copy image to: [inpaint sketch]\" in some img2img's tab,\n" +
        "if scale<100% the canvas will be invisible,\n" +
        "else if scale>100% this page will take large amount of memory and CPU performance.";
    div.appendChild(note2);

    var btn = document.createElement("button");
    btn.innerHTML = "Click here to ignore";

    div.appendChild(btn);

    function tryShowTopBar(scale) {
        if (showing) return;

        b.innerHTML = 'Bad Scale: ' + scale + '% ';

        var updateScaleTimer = setInterval(function() {
            var newScale = getScale();
            b.innerHTML = 'Bad Scale: ' + newScale + '% ';
            if (newScale == 100) {
                var p = div.parentNode;
                if (p != null) p.removeChild(div);
                showing = false;
                clearInterval(updateScaleTimer);
                check();
            }
        }, 999);

        btn.onclick = function() {
            clearInterval(updateScaleTimer);
            var p = div.parentNode;
            if (p != null) p.removeChild(div);
            ignore = true;
            showing = false;
            localStorage.setItem("bad-scale-ignore-it", "ignore-it");
        };

        document.body.appendChild(div);
    }

    function check() {
        if (!ignore) {
            var timer = setInterval(function() {
                var scale = getScale();
                if (scale != 100 && !ignore) {
                    tryShowTopBar(scale);
                    clearInterval(timer);
                }
                if (ignore) {
                    clearInterval(timer);
                }
            }, 999);
        }
    }

    if (document.readyState != "complete") {
        document.onreadystatechange = function() {
            if (document.readyState != "complete") check();
        };
    } else {
        check();
    }
})();
