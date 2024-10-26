const container1 = document.querySelector(".container1");
const sections = gsap.utils.toArray(".container1 section");
const texts = gsap.utils.toArray(".anim");
const mask = document.querySelector(".mask");
const initialPosition = -50 * (sections.length - 1) / 2;

let scrollTween = gsap.to(sections, {
  xPercent: -100 * (sections.length - 1),
  ease: "none",
  scrollTrigger: {
    trigger: ".container1",
    pin: true,
    scrub: 1,
    end: "+=3000",
    onUpdate: self => {
      // Update pin spacing manually
      self.pinSpacing = container1.scrollWidth - window.innerWidth;
    },
    onEnter: () => {
      container1.classList.add("horizontal-scrolling"); // Add a class while horizontal scrolling is active
    },
    onLeaveBack: () => {
      container1.classList.remove("horizontal-scrolling"); // Remove the class when horizontal scrolling ends
    }
  }
});

// Progress bar animation

gsap.to(mask, {
  width: "105%",
  scrollTrigger: {
    trigger: ".wrapper",
    start: "top left",
    scrub: 1
  }
});

// Pin the element "trust_customers" during horizontal scrolling
gsap.to(".trust_customers", {
  scrollTrigger: {
    trigger: ".container1",
    start: "top top",
    end: "+=3000",
    scrub: 1,
    onUpdate: self => {
      self.pinSpacing = container1.scrollWidth - window.innerWidth;
    },
    onEnter: () => {
      if (container1.classList.contains("horizontal-scrolling")) {
        // Only pin if horizontal scrolling is active
        container1.classList.add("trust_customers-pinned");
      }
    },
    onLeaveBack: () => {
      container1.classList.remove("trust_customers-pinned");
    }
  }
});
