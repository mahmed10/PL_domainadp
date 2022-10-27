FROM ufoym/deepo:all-jupyter-py36-cu111
LABEL maintainer="Masud Ahmed <mahmed10@umbc.edu>"

RUN pip install --upgrade pip
RUN pip --no-cache-dir install keras

EXPOSE 8443

RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN mkdir -p /root/.code-server/extensions

RUN wget https://github.com/microsoft/vscode-python/releases/download/2020.5.86806/ms-python-release.vsix
RUN code-server --install-extension ms-python-release.vsix

RUN mkdir -p /root/.local/share/code-server/User/
COPY settings.json /root/.local/share/code-server/User/
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir='/notebooks'"]
