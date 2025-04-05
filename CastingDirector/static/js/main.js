let allStars = [];
const dropZones = document.querySelectorAll(".drop-zone");
const ratingDisplay = document.getElementById("rating-value");

// Load all stars
fetch("/static/stars.json")
  .then(res => res.json())
  .then(data => {
    allStars = data;
  });


// On drag start, store all necessary info in dataTransfer
function dragStart(e) {
  const card = e.currentTarget;
  e.dataTransfer.setData("text/plain", card.dataset.name);
  e.dataTransfer.setData("image", card.dataset.image);
  e.dataTransfer.setData("label", card.dataset.label);
}

// Handle drop
function handleDrop(e) {
  e.preventDefault();
  const name = e.dataTransfer.getData("text/plain");
  const imageUrl = e.dataTransfer.getData("image");
  const label = e.dataTransfer.getData("label");

  // Update the drop zone
  e.currentTarget.innerHTML = `
    <img src="${imageUrl}" alt="${name}" style="width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 6px;" onerror="this.src='/static/default.jpg'"/>
    <div style="text-align: center; margin-top: 4px;">
      <strong>${name}</strong><br/>
      <em>${label}</em>
    </div>
  `;
  e.currentTarget.setAttribute("data-name", name);

  tryPredict();
}

// Setup all drop zones
dropZones.forEach(zone => {
  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("hover");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("hover");
  });

  zone.addEventListener("drop", e => {
    zone.classList.remove("hover");
    handleDrop(e);
  });
});

// Make prediction if all names are filled
function tryPredict() {
  const names = Array.from(dropZones).map(zone => zone.dataset.name);
  // if (names.some(name => !name)) return;

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ names }),
  })
    .then(res => res.json())
    .then(data => {
      if (data.rating) {
        ratingDisplay.textContent = `${data.rating} / 100`;
      } else {
        ratingDisplay.textContent = "Error";
      }
    })
    .catch(() => {
      ratingDisplay.textContent = "Prediction failed";
    });
}


// Toggle Info Drawer
const hamburger = document.getElementById("hamburger");
const infoDrawer = document.getElementById("infoDrawer");
document.getElementById('reset-button').addEventListener('click', () => {
  const defaultLabels = {
    "Star1": "Star 1",
    "Star2": "Star 2",
    "Star3": "Star 3",
    "Star4": "Star 4",
    "Director": "Director"
  };

  const dropZones = document.querySelectorAll('.drop-zone');

  dropZones.forEach(zone => {
    // Clear all children
    while (zone.firstChild) {
      zone.removeChild(zone.firstChild);
    }

    // Get the role from data-role
    const role = zone.getAttribute('data-role');
    const defaultLabel = defaultLabels[role] || "Slot";

    // Recreate the caption element
    const caption = document.createElement('div');
    caption.classList.add('drop-text');
    caption.textContent = defaultLabel;

    // Append the caption back into the drop-zone
    zone.appendChild(caption);
  });

  // Reset prediction text
  document.getElementById('rating-value').textContent = '--';
});
