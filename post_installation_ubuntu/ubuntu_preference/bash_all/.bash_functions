# customized shell functions

function setgit() {
  git config credential.helper store
  git push
  git config --global user.email wubinbai@yahoo.com
}
