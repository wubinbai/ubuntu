alias a='ls -a'
alias b='cd ~/trash'
alias c='cd'
alias d='du -sh *'
alias e='evince'
alias f='df -h .'
alias g='gedit'
alias h='help'
alias i='ipython3 -i'
alias j='jupyter notebook'
alias k='docker'
# no l
alias m='mkdir'
alias n='nautilus .'
alias o='cd ~/Documents'
alias p='git push'
alias q='git pull'
#alias r='rm -i'
alias s='sudo'
alias t='tar'
alias u='sudo apt update'
alias v='vim'
###???alias w='' what is w mean originally?
alias x='pwd | xsel -ib'
alias y='date'
alias z='cal'

# =================================
alias A=''
alias B='ls -AFlS' # list all sort by size
alias BR='ls -AFlSr' # above, reverse by size	
alias C=''
alias D='cd ~/Downloads'
# mEdia
alias E='cd /media/*/'
alias F=''
alias G=''
alias H=''
alias I='sudo apt install'
alias J=''
alias K=''
alias L='ipython3 nbconvert --to html'
alias M='cd ~/Music'
alias N='watch nvidia-smi'
alias O=''
alias P='cd ~/Pictures'
alias Q=''
# Reverser turn: turn previous
alias R='cd -'
alias RE='retext'
alias RE2='grip'
alias S='cd /media/*/PSSD'
alias T='cd /media/*/TOSHIBA\ EXT\/'
alias T2='cd /media/*/TOSHIBA\ EXT1\/'
# U turn: turn upper level
alias U='cd ..'
alias V='cd ~/Videos'
alias W='ls | wc -l'
alias X=''
alias Y='ipython3 nbconvert --to python'
alias Z=''
# =================================
alias sdn='shutdown now'
alias rb='reboot'
alias dh='du -h .'
alias da='du -h .;du -sh *'
alias st='speedtest'
alias ST='hdparm -Tt' # disk speed e.g. ST /dev/sda
alias p3='python3'
alias ca='cat'
# === more ls ===
alias ds='ls -d */' # list all directories
alias dots="ls -A | egrep '^\.'" # list all hidden
alias dt='ls -lhAF' # list all details
alias pa='ls -AF | grep' # list pattern
alias fp='ls -d `pwd`/*' # list full paths
alias fpp='ls -d `pwd`/* | grep' # list full paths w/t pattern
alias at='ls -Alt' # list 'a'll long sort by 't'ime 
alias atr='ls -Altr' # above, reverse time
alias ltr='ls -lhtr'
alias lt='ls -lht'

# === github ===
alias cl='git clone'
alias ga='git add'
alias cm='git commit -m'
alias gm='git mv'
alias gs='git status'
alias gl='git log'
alias gd='git diff'
alias ch='git checkout'

# === docker ===
alias sdp='sudo docker ps'
alias sdr='sudo docker rm'
alias sdri='sudo docker rmi'
# === misc ===
alias sud='sudo updatedb' # updatedb for locate
alias wea='curl wttr.in'
alias ret='retext'
# === pip sources ===
alias p3t='pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/' # t: tsinghua
alias p3u='pip3 install -i http://pypi.mirrors.ustc.edu.cn/simple/' # u: ustc
alias p3d='pip3 install -i http://pypi.douban.com/simple/' # d: douban
alias p3u='pip3 install -i http://pypi.mirrors.ustc.edu.cn/simple/' # u: ustc
alias p3h='pip3 install -i http://pypi.hustunique.com/' # h: huazhongkeji
alias p3a='pip3 install -i http://mirrors.aliyun.com/pypi/simple/' # a: aliyun

# alias sduhmd="sudo du -h --max-depth=1"

# === tar ===
alias tv='tar -tzvf' # tar view .tar.gz
alias tn='tar -czvf' # tar create 'n'ew .tar.gz
alias tx='tar -xzvf' # tar extract .tar.gz

alias TV='tar -tvf' # tar view .tar
alias TN='tar -cvf' # tar create 'n'ew .tar
alias TX='tar -xvf' # tar extract .tar
# === chmod ===
alias ax='chmod a+x'
# === cd ===
alias d1='cd ..'
alias d2='cd ..; cd ..'
alias d3='cd ..; cd ..; cd ..'
alias d4='cd ..; cd ..; cd ..; cd ..'
alias d5='cd ..; cd ..; cd ..; cd ..'
alias d6='cd ..; cd ..; cd ..; cd ..; cd ..'

alias u1='cd /$(echo $PWD | cut -f 2 -d /)'
alias u2='cd /$(echo $PWD | cut -f 2,3 -d /)'
alias u3='cd /$(echo $PWD | cut -f 2,3,4 -d /)'
alias u4='cd /$(echo $PWD | cut -f 2,3,4,5 -d /)'
alias u5='cd /$(echo $PWD | cut -f 2,3,4,5,6 -d /)'

# === cp ===
alias CP='rsync -ah --progress'
