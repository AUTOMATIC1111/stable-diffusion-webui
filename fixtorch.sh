#  fixtorch.sh

echo "--- a/functional.py    2022-10-14 05:28:39.000000000 -0400
+++ b/functional.py    2022-10-14 05:39:25.000000000 -0400
@@ -2500,7 +2500,7 @@
         return handle_torch_function(
             layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
         )
-    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
+    return torch.layer_norm(input.contiguous(), normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
 
 
 def group_norm(
" | patch -p1 -d "$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")"/nn
