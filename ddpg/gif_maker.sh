files=$(ls */*critic.png | sort -n); convert -delay 40 $files critic.gif 
#files=$(ls */*traj.png | sort -n); convert -delay 40 $files trajs.gif
