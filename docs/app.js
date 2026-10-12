'use strict';
const tasks = [
  {id:'redtoy',title:'1D Moving Robot',group:'dynamic',kind:'Dynamic manipulation',description:'The target moves along one axis while the robot attempts to catch and lift it.'},
  {id:'greentoy',title:'2D Moving Robot',group:'dynamic',kind:'Dynamic manipulation',description:'The robot responds to target motion that changes across two axes during execution.'},
  {id:'dropball',title:'Drop Ball to Cup',group:'dynamic',kind:'Dynamic manipulation',description:'The robot must time its release to place a ball into a moving cup.'},
  {id:'catchblock',title:'Catch Block',group:'unseen',kind:'Unseen object',description:'A different target object tests whether the learned feedback transfers beyond the training object.'},
  {id:'redtoyspeed',title:'1D Robot Speed',group:'unseen',kind:'Unseen speed',description:'The moving target follows a speed range not seen during training.'},
  {id:'dropballcolor',title:'Drop Ball Color',group:'unseen',kind:'Unseen color',description:'A change in cup color tests visual generalization during dynamic placement.'}
];
const tabs = document.getElementById('task-tabs');
const videos = [document.getElementById('ours-video'), document.getElementById('baseline-video')];
const playButton = document.getElementById('play-both');
const status = document.getElementById('playback-status');
let selected = 'redtoy';
let activeFilter = 'all';
let revision = 0;

for (const task of tasks) {
  const button = document.createElement('button');
  button.className = 'task-tab';
  button.id = 'task-' + task.id;
  button.type = 'button';
  button.setAttribute('role', 'tab');
  button.setAttribute('aria-controls', 'comparison');
  button.dataset.id = task.id;
  const image = document.createElement('img');
  image.src = `assets/posters/${task.id}_s.jpg`;
  image.alt = '';
  image.loading = 'lazy';
  const caption = document.createElement('span');
  caption.textContent = task.title;
  const sub = document.createElement('small');
  sub.textContent = task.kind;
  caption.append(sub);
  button.append(image, caption);
  button.addEventListener('click', () => selectTask(task.id));
  button.addEventListener('keydown', event => {
    const visible = [...tabs.children].filter(x => !x.hidden);
    const i = visible.indexOf(button);
    let target;
    if (event.key === 'ArrowRight') target = visible[(i + 1) % visible.length];
    if (event.key === 'ArrowLeft') target = visible[(i - 1 + visible.length) % visible.length];
    if (event.key === 'Home') target = visible[0];
    if (event.key === 'End') target = visible.at(-1);
    if (target) { event.preventDefault(); target.focus(); selectTask(target.dataset.id); }
  });
  tabs.append(button);
}

function updatePlaybackButton() {
  const playing = videos.some(video => !video.paused && !video.ended);
  playButton.textContent = playing ? 'Ⅱ Pause both' : '▶ Play both';
}

function selectTask(id) {
  const task = tasks.find(x => x.id === id);
  if (!task) return;
  revision++;
  selected = id;
  for (const button of tabs.children) {
    const active = button.dataset.id === id;
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
  document.getElementById('comparison').setAttribute('aria-labelledby', 'task-' + id);
  document.getElementById('task-title').textContent = task.title;
  document.getElementById('task-category').textContent = task.kind.toUpperCase();
  document.getElementById('task-description').textContent = task.description;
  document.getElementById('task-index').textContent = String(tasks.indexOf(task) + 1).padStart(2, '0') + ' / 06';
  ['s', 'f'].forEach((variant, i) => {
    const video = videos[i];
    video.pause();
    video.poster = `assets/posters/${id}_${variant}.jpg`;
    video.src = `assets/videos/${id}_${variant}.mp4`;
    video.load();
    video.setAttribute('aria-label', `${i === 0 ? 'LoopAct' : 'GR00T'}: ${task.title}`);
    document.getElementById(i === 0 ? 'ours-download' : 'baseline-download').href = video.src;
  });
  status.textContent = 'Start together. Each clip keeps its original timing.';
  updatePlaybackButton();
}

async function startBoth() {
  const currentRevision = revision;
  try {
    for (const video of videos) if (video.ended) video.currentTime = 0;
    await Promise.all(videos.map(video => video.play()));
    if (currentRevision === revision) status.textContent = 'Playing both original clips. The shorter clip holds its last frame.';
  } catch (error) {
    if (currentRevision !== revision) return;
    videos.forEach(video => video.pause());
    status.textContent = 'Playback could not start together. Please use the individual video controls.';
  }
  updatePlaybackButton();
}

playButton.addEventListener('click', () => {
  if (videos.some(video => !video.paused && !video.ended)) {
    videos.forEach(video => video.pause());
    status.textContent = 'Both clips paused.';
  } else startBoth();
});
document.getElementById('restart').addEventListener('click', () => {
  videos.forEach(video => { video.pause(); video.currentTime = 0; });
  startBoth();
});
for (const video of videos) {
  for (const event of ['play', 'pause', 'ended']) video.addEventListener(event, updatePlaybackButton);
  video.addEventListener('error', () => { status.textContent = 'A clip could not be loaded. Try its download link below the video.'; });
}
for (const button of document.querySelectorAll('[data-filter]')) {
  button.addEventListener('click', () => {
    activeFilter = button.dataset.filter;
    for (const other of document.querySelectorAll('[data-filter]')) {
      const active = other === button;
      other.classList.toggle('active', active);
      other.setAttribute('aria-pressed', String(active));
    }
    for (const tab of tabs.children) {
      const task = tasks.find(x => x.id === tab.dataset.id);
      tab.hidden = activeFilter !== 'all' && task.group !== activeFilter;
    }
    const current = tasks.find(x => x.id === selected);
    if (activeFilter !== 'all' && current.group !== activeFilter) selectTask(tasks.find(x => x.group === activeFilter).id);
  });
}
selectTask(selected);
