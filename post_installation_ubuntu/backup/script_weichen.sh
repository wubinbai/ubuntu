for item in .bashrc .bash_history .zshrc .zsh_history .gitconfig .vimrc .tmux.conf .jupyter .theanorc .ssh; do
	cp -r $HOME/${item} ${BACKUP_LOC}/desktop_home_dir/
done
