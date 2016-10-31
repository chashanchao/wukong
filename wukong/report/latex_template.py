"""
    Latex templates for generating reports 
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

import copy

doc_pre_template=r"""
\documentclass[11pt]{article}
\usepackage{latexsym,amssymb,amsmath} % for \Box, \mathbb, split, etc.
% \usepackage[]{showkeys} % shows label names
\usepackage{cite} % sorts citation numbers appropriately
\usepackage{path}
\usepackage{url}
\usepackage{verbatim}
\usepackage[pdftex]{graphicx}

\usepackage{subfigure}
\usepackage{graphicx}
\usepackage[flushleft]{threeparttable}
%\usepackage[commentsnumbered]{algorithm2e}
\usepackage[ruled,vlined,commentsnumbered,linesnumbered]{algorithm2e}

\usepackage{rotating}

\usepackage{tabulary}
\usepackage{booktabs}
\usepackage{amsmath, amsthm, amssymb}

% FONTS
%\usepackage{fourier}
%\usepackage{charter}
%\usepackage{avant}
%\usepackage{bookman}
%\usepackage[garamond]{mathdesign}
%\usepackage{helvet}
%\usepackage[palatino,gill,courier]{altfont}
%\usepackage{pxfonts}
%\usepackage{newcent}
%\usepackage[scaled]{uarial}
%\usepackage{times}
%\renewcommand{\familydefault}{\sfdefault}
%\usepackage{arev}
%\usepackage[charter]{mathdesign}
%\usepackage{berasans}
%\usepackage{beraserif}
%\usepackage{palatino}
\usepackage{mathpazo}


% THEOREM
\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem*{remark}{Remark}

% TABLE
\usepackage{array}

% horizontal margins: 1.0 + 6.5 + 1.0 = 8.5
\setlength{\oddsidemargin}{0.0in}
\setlength{\textwidth}{6.5in}
% vertical margins: 1.0 + 9.0 + 1.0 = 11.0
\setlength{\topmargin}{0.0in}
\setlength{\headheight}{12pt}
\setlength{\headsep}{13pt}
\setlength{\textheight}{625pt}
\setlength{\footskip}{24pt}

\renewcommand{\textfraction}{0.10}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\floatpagefraction}{0.90}

\makeatletter
\setlength{\arraycolsep}{2\p@} % make spaces around "=" in eqnarray smaller
\makeatother

% change equation, table, figure numbers to be counted inside a section:
\numberwithin{equation}{section}
\numberwithin{table}{section}
\numberwithin{figure}{section}

% set two lengths for the includegraphics commands used to import the plots:
\newlength{\fwtwo} \setlength{\fwtwo}{0.45\textwidth}
% end of personal macros

\begin{document}
"""

doc_post_template = r""" \end{document} """


table_template = r"""
\begin{table}[!htp]
    \scriptsize
    \centering
    \caption{ MY_TABLE_CAPTION }
    \begin{threeparttable}
        \begin{tabular}[t]
        TABLE_DATA  
        \end{tabular}
    \end{threeparttable}
\end{table}
"""


figure_template = r"""
\begin{figure}[!htp]
    \centering
    \includegraphics[width= MY_FIGURE_WIDTH \textwidth]
        {MY_FIGURE_FILE}
    \caption{MY_FIGURE_CAPTION}
    \label{My_FIGURE_LABEL}
\end{figure}
"""


def fill_latex_table(table_data=None, 
                     caption="My Table"):

    # Check parameters
    filled_table = copy.copy(table_template)
    filled_table = filled_table.replace("MY_TABLE_CAPTION", caption)
    
    n_rows = len(table_data)
    n_columns = len(table_data[0])   
    table_data_str = '{' + '|c'*n_columns + '|} \\hline \n'
    for i in range(n_rows): #row
        table_data_str += table_data[i][0]
        for j in range(1, n_columns): #row
            table_data_str += ('& ' + table_data[i][j])
            
        table_data_str += ' \\\ \\hline \n'
   
    filled_table = filled_table.replace('TABLE_DATA', table_data_str)
    return filled_table

    
def create_figure(figure=None,
                  width=1.0,
                  caption="My Figure",
                  label="my_figure_label"):
    tex_figure = copy.copy(figure_template)
    tex_figure = tex_figure.replace('MY_FIGURE_CAPTION', caption)
    tex_figure = tex_figure.replace('MY_FIGURE_FILE', figure)
    tex_figure = tex_figure.replace('My_FIGURE_LABEL', label)
    tex_figure = tex_figure.replace('MY_FIGURE_WIDTH', str(width))
    #tex_figure = tex_figure.replace('My_FIGURE_HEIGHT', str(height))
    return tex_figure

















