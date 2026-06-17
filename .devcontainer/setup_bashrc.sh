#!/bin/bash
MARKER="# === Custom PS1 / Git Branch ==="

# Only append if not already present (idempotent)
if ! grep -qF "$MARKER" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# === Custom PS1 / Git Branch ===
parse_git_branch() {
    local branch
    branch=$(git branch 2>/dev/null | grep '^\*' | sed 's/\* //')
    [ -n $"$branch" ] && echo "($branch)"
}
PS1_date="\[\e[38;2;203;166;247m\]\d\[\e[0m\]\[\e[38;2;205;214;244m\]"
PS1_time="\[\e[0m\]\[\e[38;2;203;166;247m\]\t\[\e[0m\]\[\e[38;2;205;214;244m\]"
PS1_host="\[\e[0m\]\[\e[38;2;116;199;236m\]@brain_state_pred\[\e[0m\]\[\e[38;2;205;214;244m\]"
PS1_wdir="\[\e[0m\]\[\e[38;2;137;180;250m\]\w"
PS1_gitbranch="\[\e[38;2;242;205;205m\]\$(parse_git_branch)"
PS1_gt="\[\e[1m\]\[\e[0m\]\[\e[38;2;250;179;135m\]>"
PS1_other="\[\e[0m\]\[\e[0m\]\[\e[38;2;205;214;244m\]"
export PS1="${PS1_date} ${PS1_time} ${PS1_host} ${PS1_wdir} ${PS1_gitbranch}${PS1_gt}${PS1_other} \[$(tput sgr0)\]\n"
EOF
  echo "PS1 customizations appended to ~/.bashrc"
else
  echo "PS1 customizations already present, skipping."
fi
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
cp /projects/bwood/classes/brain_state_pred/.devcontainer/devcontainer.bash_aliases ~/.bash_aliases
cp -r /projects/bwood/classes/brain_state_pred/.devcontainer/dotfiles/.config ~/
curl -LsSf https://astral.sh/uv/install.sh | sh