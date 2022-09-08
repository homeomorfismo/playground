# How to set up a PSU site

1. Install Jekyll! It is way easier. First install Ruby, then modify the `GEM_HOME` variable and include the `gems/bin` directory into `PATH`. Then install and create a site.
```
sudo apt-get install ruby-full build-essential zlib1g-dev
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
gem install jekyll bundler
mkdir $HOME/personal-sites
cd $HOME/personal-sites
jekyll new <site-name>
cd $HOME/personal-sites/<site-name>
bundle add webrick
```
2. Modify your site. You might want to remove the `_posts` directory if your aim is a simple static site (no blog, no posts). Edit the main files, add your stuff. Main sites goes into the root directory of the site. You can organize your site in a more fancy way. Read and google a Jekyll tutorial.
Do `bundle exec jekyll serve` and check the site is fine.

3. Put it online!
```
sftp <ODIN>@odin.pdx.edu
<password>
lcd /_site
cd public-html 
put -fR *
```
