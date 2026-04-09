# CopyIfMissing.cmake
# Copies FILE to DEST only if DEST does not already exist.
# This preserves user edits to moddable files across rebuilds.
#
# Usage (from add_custom_command):
#   ${CMAKE_COMMAND} -DFILE=<src> -DDEST=<dst> -P CopyIfMissing.cmake
if(NOT EXISTS "${DEST}")
    file(COPY_FILE "${FILE}" "${DEST}")
    message(STATUS "[CopyIfMissing] Installed ${DEST}")
endif()
