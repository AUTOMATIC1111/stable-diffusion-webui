npm -v
8.19.2

node -v
v19.0.0

git -v
git version 2.38.1.windows.1

## GIT 
File > Close folder
Source Control -> Clone -> Paste:
git clone https://bitbucket.org/ev45ive/technikum-polna-angular technikum-polna-angular
cd technikum-polna-angular
npm i
ng serve -o 

## GIT Update
git stash -u 
git pull -f 

## GIT Stash
git stash list
git checkout <commit>
git stash apply <stashId>

## Environment Variables
echo $PATH
%USERPROFILE%\AppData\Roaming\npm
npm i -g http-server
where http-server 
http-server

## NPM 
%USERPROFILE%\.npmrc
prefix = "C:\\Users\\  {wasz username} \\AppData\\Roaming\\npm"

npm config get prefix
npm config set prefix C:\Users\{username}\AppData\Roaming\npm

npm run nazwa_skryptu
npm run ng serve ...
npm run ng generate ...

## Angular CLI
npm i -g @angular/cli 

ng help
ng version
Angular CLI: 14.2.6

ng new --help
ng new szkolenie-angular
Would you like to add Angular routing? Yes
? Which stylesheet format would you like to use? SCSS   
[https://sass-lang.com/documentation/syntax#scss 

ng serve -o 

## Extensions
Angular Language Service
Angular 10 Snippets - Mikael Morlund

## Angular DevTools
https://angular.io/guide/devtools

## Angular Schematics (generatory)
ng g --help
ng g m --help

## UI Toolkits
https://material.angular.io/components/menu/overview
https://ng-bootstrap.github.io/#/components/accordion/examples
https://ng.ant.design/docs/introduce/en
https://www.primefaces.org/primeng-v8-lts/#/
https://www.telerik.com/kendo-angular-ui/components/grid/
https://js.devexpress.com/Overview/Angular/


## Playlists module schematics
ng g m playlists -m app --route playlists --routing 

ng g c playlists/containers/playlists-view

ng g c playlists/components/playlist-list
ng g c playlists/components/playlist-details
ng g c playlists/components/playlist-editor


## Music module schematics
ng g m music -m app --route music --routing 

ng g c music/containers/album-search-view
ng g c music/containers/album-details-view

ng g c music/components/search-form
ng g c music/components/results-grid
ng g c music/components/album-card

ng g s core/services/music-api
ng g interface core/model/Album


ng g c shared/containers/page-not-found



## Bootstrap CSS
npm install bootstrap@5.2.2

## Shared module
ng g m shared -m playlists

ng g m core -m app

## Fake Data generators
https://fakerjs.dev/api/
casual.js
https://pravatar.cc/
https://www.uifaces.co/
https://www.placecage.com/


## Spotify
holoyis165@bulkbye.com
placki777

## Fork GIT repo

https://bitbucket.org/ev45ive/technikum-polna-angular/src/master/ -> fork !

Ctrl Shift + P -> Clone 

git clone https://bitbucket.org/_WASZ_LOGIN_A_NIE_MOJ_/technikum-polna-angular/ 
File -> Open Folder
npm i 
npm start

git remote 
git remote show origin 
git remote add trener https://bitbucket.org/ev45ive/technikum-polna-angular/

## Config Author 
git config --global user.name "Mateusz Kulesza"
git config --global user.email "Mateusz@Kulesza.zz"

## Pull changes
git switch master
git pull -u trener master

## Push Changes
git switch -c nazwa_galezi
git status

/// Do some work... Save files...

git add .
git commit -m "Opisujemy co zrobilismy"

git push origin nazwa_galezi

>> Na stronie -> Create pull request (tylko raz!, potem tylko push!)
