<%!
import version
%>

# Use nvidia tensorflow image
FROM nvcr.io/nvidia/tensorflow:19.10-py3


# Add repo version to image as last step
RUN echo ${version.Git.pretty} > /git_version.txt

