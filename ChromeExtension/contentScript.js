/*
  File: contentScript.js
  Version: 1.0
  Description: Content script for a Chrome extension. Listens for messages from the popup script to activate image scanning functionality on web pages.
  Communicates back with the background script to signal completion or cancellation of the scan.
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
*/

// Listen for messages from popup script
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  if (message.type === "activateImageScan") {
    // Activate listener for click on images
    document.addEventListener("click", imageClickListener, { once: true });
  }
});

function imageClickListener(event) {
  if (event.target.tagName === "IMG") {
    var imageUrl = event.target.src;
    chrome.runtime.sendMessage({type: "imageClicked", url: imageUrl});

    chrome.runtime.sendMessage({type: "imageScanComplete"});
  } else {
    console.log("Clicked element is not an image:", event.target.tagName);
    // Only send the cancelled message if clicked outside an image
    chrome.runtime.sendMessage({type: "imageScanCancelled"});
  }
  // Remove the listener
  document.removeEventListener("click", imageClickListener);
}
