let video;
let poseNet;
let poses = [];

function setup() {
  // Create a canvas and attach it to the body
  const canvas = createCanvas(640, 480);
  canvas.parent(document.body); // Attach the canvas to the body element

  // Access the webcam
  video = createCapture(VIDEO, () => {
    console.log("Webcam initialized!");
  });
  video.size(width, height);
  video.hide(); // Hide the default video element

  // Load the PoseNet model
  console.log("Loading PoseNet model...");
  poseNet = ml5.poseNet(video, modelReady);

  // Listen for pose detections
  poseNet.on("pose", results => {
    poses = results;
  });
}

function modelReady() {
  console.log("PoseNet model ready!");
}

function draw() {
  // Draw the webcam feed on the canvas
  image(video, 0, 0, width, height);

  // Draw the keypoints and skeleton if poses are detected
  if (poses.length > 0) {
    drawKeypoints();
    drawSkeleton();
  } else {
    // Display a message if no poses are detected
    fill(255, 0, 0);
    textSize(16);
    textAlign(CENTER, CENTER);
    text("No body detected", width / 2, height - 20);
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < poses.length; i++) {
    const pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      const keypoint = pose.keypoints[j];
      if (keypoint.score > 0.2) { // Only draw keypoints with a confidence score above 0.2
        fill(0, 255, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeleton
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