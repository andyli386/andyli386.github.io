#!/bin/bash
cd $BLOGHOME
cp themes/next/_config.yml source/ori_data/themes/next/config.yml
cp _config.yml source/ori_data/config.yml
cp source/_posts/* source/ori_data/posts/
cp scaffolds/* source/ori_data/scaffolds/
cd $BLOGHOME/source/
git add .
git commit -m "Update blog"
git push origin master
hexo g && hexo d
