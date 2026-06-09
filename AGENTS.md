<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current Spec Kit plan
for the active numbered feature. For this consolidation branch, use
docs/MERGE_REVIEW.md and docs/PROJECT_INVENTORY.md as the current workflow
context. The active development baseline is devControl.

SDRwatch is operated through the web GUI during normal use and user acceptance
testing. The scanner CLI is an internal backend entrypoint invoked by the
controller/service layer; use CLI checks only as scanner backend smoke tests, not
as the required operator workflow. Future operator-facing features must expose
their workflow through the web UI and controller job lifecycle.
<!-- SPECKIT END -->
