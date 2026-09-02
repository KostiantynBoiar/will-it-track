% =============================================================================
% mproj.cls
%
% Template for masters year projects in the School of Computing Science at the
% University of Glasgow. Adapted from the level 3 project template originally
% provided by Colin Perkins <csp@dcs.gla.ac.uk>.
%
% =============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mproj}[%
2009/01/09 Colin Perkins <csp@dcs.gla.ac.uk>.%
2009/09/16 Modified by Tim Storer <tws@dcs.gla.ac.uk> to support non pdf modes.%
2011/02/15 Modified by Gregg Hamilton <hamiltgr@dcs.gla.ac.uk> for use in final year projects.%
2012/02/21 Modified by Gethin Norman <gethin.norman@glasgow.ac.uk> for use in masters projects and proposals.%
2021/11/17 Modified by Mark McGill <mark.mcgill@glasgow.ac.uk> to bring the latex template close in parity to the MSWord template (same margins, font, roughly same heading sizes)
]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[11pt,a4paper]{report}

% Changes from MMG to bring the template in line with the MSWord equivalent. Changed the section title spacing and font to match.
\usepackage{titlesec}
\titlespacing\chapter{0pt}{0pt plus 2pt minus 2pt}{24pt plus 2pt minus 2pt}
\titlespacing\section{0pt}{0pt plus 2pt minus 2pt}{0pt plus 2pt minus 2pt}
\titlespacing\subsection{0pt}{0pt plus 2pt minus 2pt}{0pt plus 2pt minus 2pt}
\titlespacing\subsubsection{0pt}{0pt plus 2pt minus 2pt}{0pt plus 2pt minus 2pt}
\titlespacing*{\paragraph} {0pt}{0ex plus 1ex minus .2ex}{1em}
\titlespacing*{\subparagraph} {\parindent}{0ex plus 1ex minus .2ex}{1em}

\titleformat{\chapter}[hang] 
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter:}{1em}{} 


\setlength{\parindent}{0pt}
\setlength{\parskip}{11pt plus 2pt}
\addtolength\textwidth{1.0in}
\addtolength\oddsidemargin{-0.50in}
\addtolength\textheight{1.0in}
\addtolength\topmargin{-0.5in}

\usepackage[a4paper,
bindingoffset=0.2in,
left=3.18cm,
right=3.18cm,
top=2.54cm,
bottom=2.54cm,
footskip=.25in]{geometry}

\usepackage{newcent}

\renewcommand\maketitle{\begin{titlepage}%
\let\footnotesize\small
\let\footnoterule\relax
\let\footnote \thanks
\ \vskip2em
\includegraphics[trim=0.7cm 0.7cm 0.7cm 0.7cm]{images/CompSci_colour}
  \centering
  \Large
  \vskip 6em%
  {\bf \Huge \@title \par}%
  \vskip 2em%
  \huge
  \begin{tabular}[t]{@{}l}%
    \@author
  \end{tabular}\par%
  \Large
  \vskip 2em%
  School of Computing Science\\
  Sir Alwyn Williams Building\\
  University of Glasgow\\
  G12 8QQ
  \Large
  \vskip 2em%
  A dissertation presented in part fulfilment of the requirements of the Degree of Master of Science at The University of Glasgow
  \vskip 2em%
  \@date 
  \vfill
\end{titlepage}%
\setcounter{footnote}{0}%
\global\let\thanks\relax
\global\let\maketitle\relax
\global\let\@thanks\@empty
\global\let\@author\@empty
\global\let\@date\@empty
\global\let\@title\@empty
\global\let\title\relax
\global\let\author\relax
\global\let\date\relax
\global\let\and\relax
}

\newcommand{\educationalconsent}{
\newpage
\section*{Education Use Consent}

I hereby give my permission for this project to be shown to other
University of Glasgow students and to be distributed in an electronic
format.  \textbf{Please note that you are under no obligation to sign 
this declaration, but doing so would help future students.}

\begin{tabular}{@{}llll}
      &                     &            &                     \\
      &                     &            &                     \\
Name: & \rule{4.0cm}{0.5mm} & Signature: & \rule{4.5cm}{0.5mm} \\
      &                     &            &                     \\
      &                     &            &                     \\
\end{tabular}
}
