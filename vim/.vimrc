filetype on
filetype plugin on 
filetype indent on
syntax on

set number
set shiftwidth=4
set tabstop=4
set expandtab

set showmode
set showmatch
set hlsearch

set history=30

set wildmenu
set wildmode=list:longest
set wildignore=*.docx,*.jpg,*.png,*.gif,*.pdf,*.pyc,*.exe,*.flv,*.img,*.xlsx

" Pluggins "

call plug#begin('~/.vim/plugged')

Plug 'tpope/vim-surround'
Plug 'preservim/nerdtree'
Plug 'vim-syntastic/syntastic'
Plug 'vim-airline/vim-airline'
Plug 'ycm-core/YouCompleteMe'

call plug#end()
