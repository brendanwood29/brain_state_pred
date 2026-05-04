#!/bin/bash
MARKER="# === Custom PS1 / Git Branch ==="

# Only append if not already present (idempotent)
if ! grep -qF "$MARKER" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# === Custom PS1 / Git Branch ===
parse_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
PS1_date="\[\033[38;5;139m\]\d\[$(tput sgr0)\]\[\033[38;5;15m\]"
PS1_time="\[$(tput sgr0)\]\[\033[38;5;139m\]\t\[$(tput sgr0)\]\[\033[38;5;15m\]"
PS1_host="\[$(tput sgr0)\]\[\033[38;5;73m\]@\h\[$(tput sgr0)\]\[\033[38;5;15m\]"
PS1_wdir="\[$(tput sgr0)\]\[\033[38;5;24m\]\w"
PS1_gitbranch="\e[38;5;204m\]\$(parse_git_branch)"
PS1_gt="\[$(tput bold)\]\[$(tput sgr0)\]\[\e[38;5;214m\]>"
PS1_other="\[$(tput sgr0)\]\[$(tput sgr0)\]\[\e[38;5;15m\]"
export PS1="${PS1_date} ${PS1_time} ${PS1_host} ${PS1_wdir}${PS1_gitbranch}${PS1_gt}${PS1_other} \[$(tput sgr0)\]\n"
EOF
  echo "PS1 customizations appended to ~/.bashrc"
else
  echo "PS1 customizations already present, skipping."
fi

cp /projects/bwood/classes/brain_state_pred/.devcontainer/devcontainer.bash_alises ~/.bash_aliases
curl -LsSf https://astral.sh/uv/install.sh | sh