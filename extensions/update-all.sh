ls | while read dir; do if [ -d "$dir/.git" ]; 
then echo "Pulling updates for $dir..."; 
git -C "$dir" pull; fi; done