# Projeto de Iniciação Científica da ufcg
# Caracterização das Cornetas dos Radiotelescópios Uirapuru e Bacurau

Este é um espaço para colaboração, observe as seguintes regras:
- Mantenha apenas versões rodando no diretório `notebooks`, versões de teste devem ficar locais e podem ser discutidas com as github issues.
- Utilize um esquema de nomes semelhante ao que esta indicado, um número de dois dígitos, as iniciais do autor e uma descrição do arquivo, evite caracteres especiais nos nomes dos arquivos.
- Todo o trabalho pode ser em português, mas os nomes de variáveis, funções, classes, etc. devem ser descrições em inglês.


1. Habilite o **subssistema do windows para o Linux** conforme as instruções em [aqui](https://www.techtudo.com.br/noticias/2016/04/como-instalar-e-usar-o-shell-bash-do-linux-no-windows-10.ghtml)
    - Clique em **INICIAR** e acesse **Configurações**, selecione **Atualizações e Segurança** e clique em **Modo de Desenvolvedor**.
    - Clique na barra de busca e digite: **Ativar e Desativar Recursos do Windows** e marque a opção **Windows Subsystem for Linux**, clique **OK**. Reinicie.
2. Instalando o Miniforge:
    - Siga os comandos abaixo, um de cada vez e leia atentamente as saídas.
    ```bash
    apt-get update
    apt-get install nano git wget build-essential
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    chmod u+x Miniforge3-Linux-x86_64.sh
    ./Miniforge3-Linux-x86_64.sh
    ```
3. Crie um ambiente conda de trabalho com o Python, para encapsular as suas necessidades de bibliotecas.
    - No **bash shell* do windows:
    ```
    conda create --name mestrado
    conda activate mestrado
    conda install astropy numpy scipy matplotlib pandas jupyter ipykernel jupyter_contrib_nbextensions
    python -m ipykernel install --user --name=mestrado
    ```

    Agora edite o arquivo `~/.bashrc` com o comando `nano ~/.bashrc` e coloque na última linha o conteúdo abaixo:
    ```
    alias jupyter-notebook="~/.local/bin/jupyter-notebook --no-browser"
    ```
    Para sair e salvar suas alterações digite `CTRL+X` e depois `Y`+ `ENTER`.

    Feche e abra o seu terminal novamente. Agora você pode abrir o `jupyter` com o comando `jupyter-notebook`. Observe que um token aparece na tela. Copie este token com o mouse, abra um navegador, aponte seu navegador para `localhjost:8888` e cole o token quando solicitado.


1. Para instalar o `miniforge` diretamente pelo windows siga as instruções em https://github.com/conda-forge/miniforge
    - Fazer o [download](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
    - Clique duas vezes e realize a instalação. Alternativamente, rode:
    ```
    start /wait "" build/Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3
    ```

**IMPORTANTE**: O que você precisa é ter um ambiente python funcionando e capaz de encapsular as suas diferentes necessidades de bibliotecas. Acredite, você vai precisar de mais do que um ambiente e você irá se deparar com o problema de ter versões incompatíveis de bibliotecas mais cedo do que tarde.

Se vopcê usar o método que se parece com o Linux, você sempre vai poder ter suporte do professor, que não tem nenhuma máquina windows para testar. As receitas publicadas seempore serão para Linux. Se você quiser manter tudo no mundo windows, tudo bem, mas não pode contar com ajuda.

# Colaboração

Colaborações são muito importantes para evoluirmos o nosso código e a participação é muito bem vinda. Antecipadamente agradecemos.

Você pode querer contribuir de várias formas diferentes, algumas delas:
- Reportar Bugs: Utilize os Issues do github.
- Consertar Bugs: Veja o que está aberto, fork, fix, push, pull request.
- Implemente novas features.
- Ajude na documentação do projeto.

 Vamos manter algumas convenções para garantir a legibilidade e usabilidade do código.

- **Desenvolva bottom-up a partir de esqueleto funcional**: Desenvolva uma função por vez, uma classe por vez, um módulo por vez. (mesmo que sua classe não faça nada ainda, ela deve poder ser importada e não dar erro nenhum.)
- **Siga boas práticas**:
     - Convenção de nomes PEP8 sempre em inglês. Considere usar `pydocstyle`.
     - Não use variáveis de um caracter.
     - Funções tipo cobra `lower_case_with_underscores`
     - Classes `CapWords`
     - Funções internas: `_single_leading(self, ...)`
     - constantes: `ALL_CAPS`
     - Documente em português
     - [PEP20](http://www.python.org/dev/peps/pep-0020/) - explícito é melhor do que implícito.
     - Cuide de erros e exceções.
     - Use templates e copie e cole código para garantir que a estrutura seja similar, principalmente as DOCSTRINGS.
- Utilize nomes de arquivos razoáveis e em pastas específicas e relevantes ao seu conteúdo. Se você não sabe onde colocar alguma coisa, crie uma pasta `scratchbooks/` e a utilize.
- **Dados são imutáveis**.
- Desenvolva de forma **determinística** e **reprodutível**.

# Fluxo de trabalho para Contribuir

## Ambiente sugerido

- Linux Ubuntu >=18.04
- python miniconda 3.9

1. Faça o fork do repositório github.
2. Siga algum processo razoável para decidir em qual parte contribuir, atendendo as demandas colocadas em ISSUES ou em contato com os desenvolvedores.
3. Faça o clone local (Mais sobre o git abaixo)

```bash
git clone git@github.com:your_name_here/radiotelescope.git
```

7. Crie um `branch` para desenvolvimento local:

```
    $ git checkout -b name-of-your-bugfix-or-feature
```

Trabalhe nos seus arquivos e divirta-se.


9. `Commit` as mudanças e envie para o github com push :

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

8. Submeta um PULL REQUEST pelo GitHub.

## Usando Git

Inicia o git no diretório:
```bash
git init
```


Adiciona arquivos no comissionamento:
```bash
git add .
```


Comissiona alterações indicadas
```bash
git commit -m "minha mensagem explicativa das alteraçoes"
# Se precisar alterar isso, use git commit --amend
```

Cria um ramo, adiciona tudo, faz o comite inicial no ramo, verificando o nome do ramo e envia para o remoto
```bash
git checkout -b my_branch_name
git add .
git commit -am "Initial commit"
git branch
git push -u origin new_branch_name
```

Alguém pode estar trabalhando nos arquivos e o master pode ter mudado, mantenha o seu branch sempre atualizado com o master.
```bash
git checkout master
git pull origin master
git checkout your_branch
git rebase master
## Resolva os conflitos que podem existir
git rebase --continue
```

- **Corrigindo problemas**

     - Fiz besteira no meu repositorio local e ele não funciona mais. Tenho uma versão funcionando em outro commit
     ```bash
     # Cuidado, é destrutivo.
     git reset --hard
     ```

     - Voltando no tempo
     ```bash
     git log
     # Ache o código do commit desejado
     git reset --hard c1fc1c2d1aa1d37c
     ```
