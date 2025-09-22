let video;
let poseNet;
let poses = [];

function setup() {
  const canvas = createCanvas(640, 480);
  canvas.parent(document.body);
  console.log("Canvas created!");

  video = createCapture(VIDEO, () => {
    console.log("Webcam initialized!");
  });
  video.size(width, height);
  video.hide();

  console.log("Loading PoseNet model...");
  poseNet = ml5.poseNet(video, modelReady);

  poseNet.on("pose", results => {
    poses = results;
  });
}

function modelReady() {
  console.log("PoseNet model ready!");
}

function draw() {
  background(220);
  image(video, 0, 0, width, height);

  if (poses.length > 0) {
    drawKeypoints();
    drawSkeleton();
  } else {
    console.log("No poses detected.");
  }
}

function drawKeypoints() {
  for (let i = 0; i < poses.length; i++) {
    const pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      const keypoint = pose.keypoints[j];
      if (keypoint.score > 0.2) {
        fill(0, 255, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

function drawSkeleton() {
  for (let i = 0; i < poses.length; i++) {
    const skeleton = poses[i].skeleton;
    for (let j = 0; j < skeleton.length; j++) {
      const partA = skeleton[j][0];
      const partB = skeleton[j][1];
      stroke(0, 255, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}