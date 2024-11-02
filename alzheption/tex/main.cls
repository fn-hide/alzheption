\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{main}

\LoadClass[a4paper,12pt]{report}
\RequirePackage[english,bahasa]{babel}
\RequirePackage{times}
\RequirePackage{setspace}
\RequirePackage{indentfirst}

\RequirePackage{arabtex}
\RequirePackage{utf8}
\setcode{utf8}

\RequirePackage{graphicx,latexsym}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{amstext}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{listings}
\RequirePackage{color}
\RequirePackage{float}
\RequirePackage{longtable}
\RequirePackage{enumerate}
\RequirePackage{multirow}
\RequirePackage[table,xcdraw]{xcolor}
\RequirePackage[center,font=footnotesize,labelfont=bf,textfont=bf,labelsep=space,aboveskip=1.5ex,singlelinecheck=off]{caption}
\RequirePackage{pdfpages}
\RequirePackage{tikz}
\usetikzlibrary{matrix,arrows}

\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000
\RequirePackage{array}
\RequirePackage{mathtools}
\RequirePackage{array}
\RequirePackage{enumitem}
\RequirePackage{caption}
\newcommand{\source}[1]{\caption*{\textmd{(Sumber: {#1})}} }
\newcommand{\tableequation}[1]{
  \vspace*{-11mm}
  {\begin{flalign}#1&&&\end{flalign}}
  \vspace*{-11mm}
}
\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}
\RequirePackage{multicol,lipsum,xparse}
\setlength{\multicolsep}{5.0pt plus 1.0pt minus 1.0pt}

\allowdisplaybreaks

% \usepackage{natbib}
% \bibliographystyle{apalike2}
\usepackage{inputenc}
\usepackage{csquotes}
\usepackage[backend=biber,natbib=true]{biblatex}
\addbibresource{references.bib}

\renewcommand{\baselinestretch}{1.75}
\parskip0.35cm

\usepackage[thinlines]{easytable}
\usepackage{tabularray}

\usepackage{CJKutf8}
% \usepackage{xeCJK}

%------------------------------------------------------------
%Layout
%------------------------------------------------------------
\setlength{\topmargin}{0cm}
\setlength{\headheight}{0.46cm}
\setlength{\headsep}{1cm}
\setlength{\topskip}{0cm}
\setlength{\oddsidemargin}{1.46cm}
\setlength{\evensidemargin}{1.46cm}
\setlength{\textwidth}{14cm}
\setlength{\textheight}{22.7cm}
\setlength{\footskip}{1.3cm}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}
\setlength{\parindent}{3em}

\pagestyle{myheadings}

\setcounter{secnumdepth}{2}
\renewcommand{\thechapter}{\Roman{chapter}}
\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}.}
\renewcommand{\thesubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.}

\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}

\usepackage{titlesec}
\titleformat{\chapter}
%
[display]
{\center\large\bfseries}
{\large\MakeUppercase{\chaptertitlename} \thechapter}{1ex}{\large}
\titlespacing{\chapter}{0cm}{-1.2cm}{2cm}

\titleformat{\section}
[hang]
{\bfseries}
{\bfseries\thesection}{1ex}{\bfseries}
\titlespacing{\section}{0cm}{3ex}{0ex}


\titleformat{\subsection}
[hang]
{\bfseries}
{\bfseries\thesubsection}{1ex}{\bfseries}
\titlespacing{\subsection}{0cm}{3ex}{0ex}

%Untuk setting penomoran teorema, lemma, definisi, dll..
\newtheoremstyle{dotlesstheorem}
  {}            % ABOVESPACE
  {0.4cm}       % BELOWSPACE
  {\itshape}    % BODYFONT
  {0pt}         % INDENT (empty value is the same as 0pt)
  {\bfseries}   % HEADFONT
  {}            % HEADPUNCT
  {  }          % HEADSPACE. `plain` default: {5pt plus 1pt minus 1pt}
  {}            % CUSTOM-HEAD-SPEC
\theoremstyle{dotlesstheorem}
\newtheorem{teorema}{Teorema}[section]
\renewcommand{\theteorema}{\arabic{chapter}.\arabic{section}.\arabic{teorema}}
\newtheorem{lemma}[teorema]{Lemma}
\newtheorem{sifat}[teorema]{Sifat}
\newtheorem{akibat}[teorema]{Akibat}
\newtheorem{proposisi}[teorema]{Proposisi}
\newtheorem{definisi}[teorema]{Definisi}

\newtheoremstyle{dotlessexample}
  {}            % ABOVESPACE
  {0.4cm}       % BELOWSPACE
  {}            % BODYFONT
  {0pt}         % INDENT (empty value is the same as 0pt)
  {\bfseries}   % HEADFONT
  {}            % HEADPUNCT
  {  }          % HEADSPACE. `plain` default: {5pt plus 1pt minus 1pt}
  {}            % CUSTOM-HEAD-SPEC
\theoremstyle{dotlessexample}
\newtheorem{contoh}[teorema]{Contoh}
\newtheorem{algoritma}{Algoritma}[chapter]
\renewcommand{\thealgoritma}{\arabic{chapter}.\arabic{algoritma}}

\renewenvironment{proof}{\vspace{-2ex}\noindent{\bf Bukti.}}{\hfill$\blacksquare$\newline\vspace{-0.4cm}}
\def\faktor#1#2{\raise0.7ex\hbox{$#1$}\!\!\!\rotatebox{-11}{$\Big/$\!}\!\lower0.7ex\hbox{$#2$}}
\def\faktorr#1#2{\raise0.4ex\hbox{$#1$}\!\!\rotatebox{-11}{$\big/$\!}\lower0.4ex\hbox{$#2$}}
\def\N{\mathbb{N}}
\def\Z{\mathbb{Z}}
\def\Q{\mathbb{Q}}
\def\R{\mathbb{R}}
\def\C{\mathbb{C}}
\def\im{\textrm{Im}}
\def\pustaka#1{\citeauthor{#1}\negthickspace,~\!\citeyear{#1}}
\def\tpustaka#1{\citeauthor{#1}~\negmedspace(\citeyear{#1})}

\usepackage[pdftex,bookmarks=true]{hyperref}


\def\captionsbahasa{
\def\prefacename{KATA PENGANTAR}
\def\contentsname{DAFTAR ISI}
\def\listfigurename{DAFTAR GAMBAR}
\def\listtablename{DAFTAR TABEL}
\def\listappendixname{DAFTAR LAMPIRAN}
\def\nomenclaturename{DAFTAR LAMBANG}
\def\abstractname{ABSTRAK}
\def\partname{BAGIAN}
\def\chaptername{BAB}
\def\appendixname{LAMPIRAN}
\def\refname{DAFTAR PUSTAKA}
\def\bibname{DAFTAR PUSTAKA}
\def\indexname{Indek}
\def\figurename{Gambar}
\def\tablename{Tabel}
\def\pagename{Halaman}
}
\def\captionsenglish{
\def\prefacename{PREFACE}
\def\contentsname{CONTENTS}
\def\listfigurename{LIST OF FIGURES}
\def\listtablename{LIST OF TABLES}
\def\listappendixname{LIST OF APPENDICES}
\def\nomenclaturename{NOMENCLATURE}
\def\abstractname{ABSTRACT}
\def\partname{PART}
\def\chaptername{CHAPTER}
\def\appendixname{APPENDIX}
\def\refname{REFERENCES}
\def\bibname{REFERENCES}
\def\indexname{Index}
\def\figurename{Figure}
\def\tablename{Table}
\def\pagename{Page}
}

%------------------------------------------------------------
%The form of page numbering
%------------------------------------------------------------
\pagestyle{myheadings}
\markright{}

\parskip0.12cm

%------------------------------------------------------------
%Define page numbering in the first chapter
%------------------------------------------------------------
\def\ps@chapterheading{%
  \let\@evenhead\@empty\let\@oddhead\@empty
  \def\@oddfoot{\hfil\thepage\hfil}%
  \def\@evenfoot{\hfil\thepage\hfil}
  }

%------------------------------------------------------------]
%Define thesis's inputs
%------------------------------------------------------------
\gdef\@university{Universitas Islam Negeri Sunan Ampel}
\gdef\@faculty{Fakultas Sains dan Teknologi}
\gdef\@dept{Sains}
\gdef\@program{Matematika}
\gdef\@city{Surabaya}
\gdef\@angkatan{2019}
\newcommand{\titleind}[1]{\gdef\@titleind{#1}}
\newcommand{\@titleind}{}
\newcommand{\titleeng}[1]{\gdef\@titleeng{#1}}
\newcommand{\@titleeng}{}
\newcommand{\fullname}[1]{\gdef\@fullname{#1}}
\newcommand{\@fullname}{}
\newcommand{\NIM}[1]{\gdef\@NIM{#1}}
\newcommand{\@NIM}{}
\newcommand{\examdate}[1]{\gdef\@examdate{#1}}
\newcommand{\@examdate}{\number\day~\ifcase\month\or
    Januari\or Pebruari\or Maret\or April\or Mei\or Juni\or
    Juli\or Agustus\or September\or Oktober\or November\or Desember\fi
    \space \number\year}
\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\@degree}{}
\newcommand{\yearsubmit}[1]{\gdef\@yearsubmit{#1}}
\newcommand{\@yearsubmit}{}
\newcommand{\headprogram}[1]{\gdef\@headprogram{#1}}
\newcommand{\@headprogram}{}
\newcommand{\firstsupervisor}[1]{\gdef\@firstsupervisor{#1}}
\newcommand{\@firstsupervisor}{}
\newcommand{\firstexaminer}[1]{\gdef\@firstexaminer{#1}}
\newcommand{\@firstexaminer}{}
\newcommand{\secondexaminer}[1]{\gdef\@secondexaminer{#1}}
\newcommand{\@secondexaminer}{}
\newcommand{\thirdexaminer}[1]{\gdef\@thirdexaminer{#1}}
\newcommand{\@thirdexaminer}{}
\newcommand{\dean}[1]{\gdef\@dean{#1}}
\newcommand{\@dean}{}

\newcommand{\semester}[1]{\gdef\@semester{#1}}
\newcommand{\@semester}{}

\newcommand{\tanggalpersetujuan}[1]{\gdef\@tanggalpersetujuan{#1}}
\newcommand{\@tanggalpersetujuan}{}

\newcommand{\nipfirstsupervisor}[1]{\gdef\@nipfirstsupervisor{#1}}
\newcommand{\@nipfirstsupervisor}{}

\newcommand{\nipfirstexaminer}[1]{\gdef\@nipfirstexaminer{#1}}
\newcommand{\@nipfirstexaminer}{}

\newcommand{\nipsecondexaminer}[1]{\gdef\@nipsecondexaminer{#1}}
\newcommand{\@nipsecondexaminer}{}

\newcommand{\nipthirdexaminer}[1]{\gdef\@nipthirdexaminer{#1}}
\newcommand{\@nipthirdexaminer}{}

\newcommand{\nipdean}[1]{\gdef\@nipdean{#1}}
\newcommand{\@nipdean}{}

%------------------------------------------------------------
%Define cover in Indonesian
%------------------------------------------------------------
\def\cover{
  \thispagestyle{empty}
  \pagenumbering{roman}
  \setcounter{page}{1}
    \begin{center}
      \begin{onehalfspacing}
      \vspace{1.2cm}	
      \MakeUppercase{\bfseries\@titleind}\par\nobreak
      \vspace{0.7cm}
            \vspace{1.5cm}
      {\normalfont \large \bfseries SKRIPSI}\\
       \vfill
       \vspace{1.0cm}
       \includegraphics[width=5cm]{img/logouinsa.png}
       \vfill
       \vspace{1.5cm}
       {\normalfont Disusun Oleh}\\
        \MakeUppercase{\normalfont\bfseries\@fullname}\\
        {\normalfont\bfseries\@NIM}\\
       \vspace{2.5cm}
       {\normalfont
         \MakeUppercase{\normalfont\bfseries Program Studi \@program}\\
         \MakeUppercase{\normalfont\bfseries\@faculty}\\
         \MakeUppercase{\normalfont\bfseries\@university}\\
         \MakeUppercase{\normalfont\bfseries\@city}\\
         \vspace{0.3cm}
         {\normalfont\bfseries\@yearsubmit}}\\
         \end{onehalfspacing}
      \end{center}
}

%------------------------------------------------------------
%Define titlepage in Indonesian
%------------------------------------------------------------
\def\titlepage{
  \chapter*{}
  \thispagestyle{empty}
  \setcounter{page}{1}
  \vspace{-3cm}
    \begin{center}
      \MakeUppercase{\normalfont\bfseries\@titleind}\par\nobreak
      \begin{onehalfspacing}
    \vfill
      {\normalfont \large \bfseries SKRIPSI}\\
      \vfill
      {\normalfont Diajukan guna memenuhi salah satu persyaratan untuk memperoleh}\\
      {\normalfont gelar Sarjana Matematika (S.Mat) pada Program Studi Matematika}\\
\vspace{1.0cm}
       \vfill
       \includegraphics[width=5cm]{img/logouinsa.png}
       \vfill
       Disusun oleh\\
       \MakeUppercase{\normalfont\bfseries\@fullname}\\
       {\normalfont\bfseries\@NIM}\\
       \vspace{2cm}
       {\normalfont
         \MakeUppercase{\normalfont\bfseries Program Studi \@program}\\
         \MakeUppercase{\normalfont\bfseries\@faculty}\\
         \MakeUppercase{\normalfont\bfseries\@university}\\
         \MakeUppercase{\normalfont\bfseries\@city}\\
         \vspace{0.5cm}
         {\normalfont\bfseries\@yearsubmit}}\\
      \end{onehalfspacing}
      \end{center}
\addcontentsline{toc}{chapter}{HALAMAN JUDUL}
}

%------------------------------------------------------------
% supervisor approval page \ lembar persetujuan pembimbing
%------------------------------------------------------------
\def\approvalpageBeforeExam{
    \newpage\thispagestyle{chapterheading}
    \phantomsection
    \begin{center}
        \MakeUppercase{\large\bfseries\expandafter{LEMBAR PERSETUJUAN PEMBIMBING}}\\
        \vspace{0.5cm}
    \end{center}

    \noindent
    Skripsi oleh\\
    \noindent
    \begin{tabular}{lllp{10cm}}
        &Nama 			& : & {\normalfont\@fullname}\\
        &NIM  			& : & {\normalfont\@NIM}\\
        &Judul Skripsi 	& : & {\normalfont\@titleind}\\
    \end{tabular}

    \vspace{0.5cm}
    \noindent
    telah diperiksa dan disetujui untuk diujikan.\\

    \begin{center}
        \singlespacing
        \begin{tabular}{p{6cm}p{9cm}}
            \begin{center}
                Pembimbing I 
            \end{center}
            
            &
            \begin{center}
                Pembimbing II
            \end{center}\\
 
            \begin{center}
                \parbox{6cm}{ \singlespacing\centering\underline {~~\@secondexaminer~~}\\ NIP. {\normalfont\@nipsecondexaminer}} \end{center} & \begin{center} \parbox{6cm}{\singlespacing\centering\underline {~~~\@thirdexaminer~~~}\\ NIP. {\normalfont\@nipthirdexaminer}} 
            \end{center} \\

            &
        \end{tabular}

        \vspace{0cm}
        \begin{tabular}{cc}
            Mengetahui,\\
            Ketua Program Studi Matematika\\
            UIN Sunan Ampel Surabaya\\
            \\
            \\
            \parbox{6cm}{ \singlespacing\centering\underline {~~~~~Yuniar Farida, M.T~~~~~}\\ NIP. 197905272014032001}\\
            &
        \end{tabular}
    \end{center}
    \addcontentsline{toc}{chapter}{LEMBAR PERSETUJUAN PEMBIMBING}
}

%------------------------------------------------------------
%Approval Page
%------------------------------------------------------------
\def\approvalpage{
  \newpage\thispagestyle{empty}
  \phantomsection
  \begin{center}
  \begin{singlespace}
  \MakeUppercase{\large\bfseries\expandafter{PENGESAHAN TIM PENGUJI SKRIPSI}}\\
  \end{singlespace}
    \end{center}
  \noindent
Skripsi oleh
  \begin{center}
  \begin{singlespace}
\begin{tabular}{lllp{8cm}}
&Nama   		& : & {\normalfont\@fullname}\\
&NIM  			& : & {\normalfont\@NIM}\\
&Judul Skripsi 	& : & {\normalfont\@titleind}
\end{tabular}

\vspace{0.4cm}
Telah dipertahankan di depan Tim Penguji\\
pada tanggal \expandafter{\@examdate}\\
\vspace{0.5cm}

Mengesahkan,\\
Tim Penguji

\singlespacing
\begin{tabular}{p{6cm}p{9cm}}
\begin{center} Penguji I \end{center} & \begin{center}Penguji II \end{center}\\
\begin{center} \parbox{6cm}{ \singlespacing\centering\underline {~~\@firstsupervisor~~}\\ NIP. {\normalfont\@nipfirstsupervisor}} \end{center} & \begin{center} \parbox{6cm}{\singlespacing\centering\underline {~~~\@firstexaminer~~~}\\ NIP. {\normalfont\@nipfirstexaminer}} \end{center} \\

\begin{center} Penguji III \end{center} & \begin{center}Penguji IV \end{center}\\
\begin{center} \parbox{6cm}{ \singlespacing\centering\underline {~~\@secondexaminer~~}\\ NIP. {\normalfont\@nipsecondexaminer}} \end{center} & \begin{center} \parbox{6cm}{\singlespacing\centering\underline {~~~~~~~~\@thirdexaminer~~~~~~~~}\\ NIP. {\normalfont\@nipthirdexaminer}} \end{center} \\
&
\end{tabular}

Mengetahui,\\
Dekan Fakultas Sains dan Teknologi\\
UIN Sunan Ampel Surabaya\\
\vspace{1.7cm}
\underline{\@dean} \\
NIP. \@nipdean 	

    \end{singlespace}
    \end{center}
\addcontentsline{toc}{chapter}{PENGESAHAN TIM PENGUJI SKRIPSI}
}

\def\approvalpagescan{
  \newpage\thispagestyle{empty}
  \phantomsection
\addcontentsline{toc}{chapter}{PENGESAHAN TIM PENGUJI SKRIPSI}
\includepdf[fitpaper]{pengesahanskripsi}
}

%------------------------------------------------------------
%Declare Page
%------------------------------------------------------------
\def\declarepage{
  \newpage\thispagestyle{chapterheading}
  \phantomsection
  \begin{center}
  {\large\bfseries\expandafter{PERNYATAAN KEASLIAN}}
  \par\nobreak
  \end{center}
  \vspace{1.0cm}
    \noindent
Saya yang bertanda tangan di bawah ini,

\noindent
\begin{tabular}{lllp{10cm}}
&Nama   & : & {\normalfont\@fullname}\\
&NIM  &  : & {\normalfont\@NIM}\\
&Program Studi & : & {\normalfont\@program}\\
&Angkatan &  : & {\normalfont\@angkatan}\\
\end{tabular}

\vspace{0.1cm}
\noindent Menyatakan bahwa saya tidak melakukan plagiat dalam penulisan skripsi saya yang berjudul "{\normalfont\@titleind}". Apabila suatu saat nanti terbukti saya melakukan tindakan plagiat, maka saya bersedia menerima sanksi yang telah ditetapkan.\\

\vspace{0.1cm}
\noindent
Demikian pernyataan keaslian ini saya buat dengan sebenar-benarnya.\\

\vspace{1.5cm}
\begin{tabular}{p{6cm}c}
&Surabaya, {\normalfont\@tanggalpersetujuan}\\
&Yang menyatakan,\\
&\\
&\\
&\parbox{5cm}{\singlespacing\centering \underline{~~~~~\@fullname~~~~~}\\ NIM. {\normalfont\@NIM}}\\
\end{tabular}
\vfill
\addcontentsline{toc}{chapter}{HALAMAN PERNYATAAN KEASLIAN}
}

\def\declarepagescan{
  \newpage\thispagestyle{chapterheading}
  \phantomsection
\addcontentsline{toc}{chapter}{HALAMAN PERNYATAAN}
\includepdf[fitpaper]{pernyataan}
}

%------------------------------------------------------------
%Motto
%------------------------------------------------------------
\def\motto{
  \chapter*{MOTTO}
\addcontentsline{toc}{chapter}{MOTTO}
}

%------------------------------------------------------------
%Acknowlegment
%------------------------------------------------------------
\def\acknowledment{
  \chapter*{HALAMAN PERSEMBAHAN}
\addcontentsline{toc}{chapter}{HALAMAN PERSEMBAHAN}
}


%------------------------------------------------------------
%Preface
%------------------------------------------------------------
\def\preface{
  \chapter*{KATA PENGANTAR}
\addcontentsline{toc}{chapter}{KATA PENGANTAR}
}

%------------------------------------------------------------
%Redefine Abstract in Indonesian
%------------------------------------------------------------
\def\abstractind{
  \clearpage\thispagestyle{chapterheading}
  \chapter*{}
  {\centering 
  {\vspace{-1.97cm}\large\textbf{\abstractname}}\\
  \vspace{1cm}
   {\textbf{\@titleind}}
        \par\nobreak
        \vskip 12\p@
    \interlinepenalty\@M
    }
  \begin{singlespace}
  \vskip 12\p@
  \addcontentsline{toc}{chapter}{\abstractname}
}
\def\endabstractind{\par\vfil\null\end{singlespace}}

%------------------------------------------------------------
%Redefine Abstract in English
%------------------------------------------------------------
\def\abstracteng{
  \clearpage\thispagestyle{chapterheading}\selectlanguage{english}
  \chapter*{}
  {\centering \normalfont
  {\vspace{-1.97cm}\large\textbf{\abstractname}}\\
  \vspace{1cm}
   {\textbf{\@titleeng}}
        \par\nobreak
        \vskip 12\p@
    \interlinepenalty\@M
    }
  \begin{singlespace}
  \vskip 12\p@
  \addcontentsline{toc}{chapter}{\abstractname}
  }
\def\endabstracteng{\par\vfil\null\end{singlespace}
\clearpage\pagenumbering{arabic}\setcounter{page}{1}}

%------------------------------------------------------------
%Nomenclature
%------------------------------------------------------------
\def\lambang{
  \clearpage\thispagestyle{chapterheading}
  \chapter*{DAFTAR LAMBANG}
  \vspace{0.5cm}
\addcontentsline{toc}{chapter}{\nomenclaturename}
}

\endinput