let capture;

function setup() {
  // Set the canvas size to 640x480 for better visibility.
  createCanvas(640, 480);

  // Create the video capture and hide the element.
  capture = createCapture(VIDEO);
  capture.hide();

  // Add a description for accessibility.
  describe('A video stream from the webcam with inverted colors.');
}

function draw() {
  // Check if the video capture is loaded.
  if (capture.loadedmetadata) {
    // Draw the video capture within the canvas.
    image(capture, 0, 0, width, width * capture.height / capture.width);

    // Invert the colors in the stream.
    filter(INVERT);
  } else {
    // Display a message if the webcam feed is not ready.
    background(0);
    fill(255);
    textSize(16);
    textAlign(CENTER, CENTER);
    text('Waiting for webcam access...', width / 2, height / 2);
  }
}