# Installation

## Recommended: Install from conda-forge

```bash
mamba install -c conda-forge imas_composer
```

This installs the package with all core dependencies (numpy, pyyaml, awkward).

## For Development

```bash
# Clone and navigate to repository
cd imas_composer

# Create environment from environment.yaml
mamba env create -f environment.yaml
mamba activate imas_composer

# Install the submodules and the package in editable mode
pip install --no-deps --no-build-isolation -e ./omas_submodule
pip install --no-deps --no-build-isolation -e ./toksearch_submodule
pip install --no-deps --no-build-isolation -e ./toksearch_d3d_submodule
pip install --no-deps --no-build-isolation -e .
```

The [environment.yaml](environment.yaml) file includes all dependencies for development and testing, including OMAS and MDSplus.

## Claude Code
Start Claude Code at the root of your checkout: in VS Code, open the checkout itself as the workspace, not a directory above it. Claude Code loads `CLAUDE.md` and `.claude/` (skills, enabled plugins) only from the directory it starts in. Started above the checkout, the rules arrive late, once Claude reads a file in it, and the skills and plugins not at all. The session hook of the `session-to-pr-skills` plugin warns you at the start of a session that is missing `CLAUDE.md`. If you work in several worktrees from a parent directory anyway, symlink `CLAUDE.md` and `.claude` of the worktree you work in into that directory.

The PR skills (`open-pr`, `improve-pr`, `propagate-stack`), `bot-token` and the session hook come from the `session-to-pr-skills` plugin (in the marketplace of the same name), shared with other repositories. The repository is private, and Claude Code cannot reliably clone private marketplaces itself, so clone it next to your imas_composer checkout with your own SSH key (you need read access to GA-FDP) and install it from there once.

Install it for your user, then switch it off for your user: `.claude/settings.json` of imas_composer switches it on again for imas_composer only, because project settings override user settings. Other projects, without a bot, never see it. Installing it for the project instead fails if `.claude` is a symlink (see above), and would write into the tracked settings file.
```
git clone git@github.com:GA-FDP/session-to-pr-skills.git
claude plugin marketplace add <absolute path to the clone>
claude plugin install session-to-pr-skills@session-to-pr-skills --scope user
claude plugin disable session-to-pr-skills@session-to-pr-skills --scope user
```
If `claude` is not on your PATH, the VS Code extension ships it under `~/.vscode-server/extensions/anthropic.claude-code-<version>/resources/native-binary/claude` (`~/.vscode/...` on a local install). Check with `claude plugin list`: run in the imas_composer checkout it shows the plugin as enabled, elsewhere as disabled.

Claude Code reads the plugin straight from the clone, so `git pull` there updates it for new sessions.

The skills run `gh`, `jq`, `openssl` and `bot-token` with `mamba run -n imas_composer`. Claude's shell is non-interactive, so `mamba` is usually not on its PATH (`conda init` only sets it up for interactive shells): say where it is in your own `~/.claude/CLAUDE.md`, e.g. "mamba is at `$HOME/miniforge3/bin/mamba`".

## GitHub access
Claude does everything on GitHub as the imas_composer bot, a GitHub App: reading PRs, issues and CI results, and opening and editing PRs through the `open-pr` skill of the `session-to-pr-skills` plugin (see "Claude Code" above). Because the bot opens PRs written by Claude, you are not their author and can review them as their first reviewer. The bot cannot merge or change repository settings. Your work is pushed with your own SSH key (`origin` is `git@github.com:GA-FDP/imas_composer.git`), so your approval never counts for it. The bot pushes only the merge commits that propagate changes through a PR stack, so that you can tell them from new work. Your own `gh` login is optional and Claude does not use it.

Each developer gets their own bot key from an organization owner, so that one key can be revoked without affecting anyone else. Treat it like a password: GA-FDP has no IP allow list, so whoever holds it can act as the bot on imas_composer from any network.
```
# Install the key where bot-token looks for it, readable only by you.
mkdir -p ~/.config/imas-composer-bot && chmod 700 ~/.config/imas-composer-bot
mv <downloaded key>.pem ~/.config/imas-composer-bot/key.pem && chmod 600 ~/.config/imas-composer-bot/key.pem

# The bot cannot tell who you are, so the open-pr skill reads your login from here
# to assign the PR to you and request your review.
git config --global github.user <your GitHub login>

# One-off check that the key works: mints a one-hour bot token, valid only for GA-FDP/imas_composer, and
# makes a read-only call. Prints GA-FDP/imas_composer. The plugin puts bot-token on the PATH of Claude's
# shell only, so ask Claude to run this. Claude mints its own token for every gh call, so GH_TOKEN never
# needs to be set permanently.
GH_TOKEN=$(mamba run -n imas_composer bot-token) mamba run -n imas_composer gh api repos/GA-FDP/imas_composer --jq .full_name
```
