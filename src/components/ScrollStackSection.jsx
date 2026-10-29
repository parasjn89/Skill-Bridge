import React, { useRef, useEffect } from 'react';

export function ScrollStackSection({ children, zIndex, bgClass = 'bg-[#10151A]', isLast = false }) {
  const outerRef = useRef(null);
  const incomingRef = useRef(null);
  const contentRef = useRef(null);

  useEffect(() => {
    let ticking = false;
    let cachedOffsetTop = null;
    let prevScale = -1;
    let prevOpacity = -1;
    let prevBrightness = -1;
    let prevInScale = -1;
    let prevInOpacity = -1;

    const measureOffset = () => {
      if (!outerRef.current) return;
      let offset = 0;
      let curr = outerRef.current;
      while (curr) {
        offset += curr.offsetTop || 0;
        curr = curr.offsetParent;
      }
      cachedOffsetTop = offset;
    };

    const update = () => {
      ticking = false;
      if (!outerRef.current) return;

      if (cachedOffsetTop === null) {
        measureOffset();
      }

      const vh = window.innerHeight;
      const top = (cachedOffsetTop !== null ? cachedOffsetTop : 0) - window.scrollY;

      // ==========================================
      // PHASE 1: OUTGOING (This section is pinned, next is sliding over)
      // The next section enters when top reaches -0.5vh (50vh pause over)
      // The next section fully covers when top reaches -1.5vh (100vh overlap)
      // ==========================================
      let targetScale = 1;
      let targetOpacity = 1;
      let targetBrightness = 1;

      if (!isLast) {
        if (top <= -0.5 * vh && top >= -1.5 * vh) {
          const p = (Math.abs(top) - 0.5 * vh) / vh; // 0 to 1
          targetScale = 1 - (p * 0.04); // 1 to 0.96
          targetOpacity = 1 - (p * 0.12); // 1 to 0.88
          targetBrightness = 1 - (p * 0.15); // 1 to 0.85
        } else if (top < -1.5 * vh) {
          targetScale = 0.96;
          targetOpacity = 0.88;
          targetBrightness = 0.85;
        }
      }

      // ==========================================
      // PHASE 2: INCOMING (This section is sliding up natively)
      // Enters at top = vh. Fully covers at top = 0.
      // ==========================================
      let targetInScale = 1;
      let targetInOpacity = 1;

      if (top > 0 && top <= vh) {
        const p = (vh - top) / vh; // 0 (at bottom) to 1 (at top)
        targetInScale = 0.98 + (p * 0.02);
        targetInOpacity = 0.94 + (p * 0.06);
      } else if (top > vh) {
        // Below screen
        targetInScale = 0.98;
        targetInOpacity = 0.94;
      }

      // ==========================================
      // APPLY DOM TRANSFORMS ONLY ON CHANGE
      // ==========================================
      if (contentRef.current && !isLast) {
        if (targetScale !== prevScale || targetBrightness !== prevBrightness) {
          contentRef.current.style.transform = `scale(${targetScale.toFixed(4)}) translateZ(-10px)`;
          contentRef.current.style.filter = `brightness(${targetBrightness.toFixed(3)})`;
          prevScale = targetScale;
          prevBrightness = targetBrightness;
        }
        if (targetOpacity !== prevOpacity) {
          contentRef.current.style.opacity = targetOpacity.toFixed(3);
          prevOpacity = targetOpacity;
        }
      }

      if (incomingRef.current) {
        if (targetInScale !== prevInScale) {
          incomingRef.current.style.transform = `scale(${targetInScale.toFixed(4)}) translateZ(0)`;
          prevInScale = targetInScale;
        }
        if (targetInOpacity !== prevInOpacity) {
          incomingRef.current.style.opacity = targetInOpacity.toFixed(3);
          prevInOpacity = targetInOpacity;
        }
      }
    };

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(update);
        ticking = true;
      }
    };

    const onResize = () => {
      measureOffset();
      onScroll();
    };

    measureOffset();
    update();

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize, { passive: true });

    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    };
  }, [isLast]);

  if (isLast) {
    return (
      <div 
        ref={outerRef} 
        className={`relative w-full ${bgClass}`} 
        style={{ zIndex, minHeight: '100vh' }}
      >
        <div ref={incomingRef} className="w-full h-full will-change-transform" style={{ minHeight: '100vh' }}>
          <div className="w-full h-full flex flex-col justify-center" style={{ minHeight: '100vh' }}>
            {children}
          </div>
        </div>
      </div>
    );
  }

  // Layout math:
  // H (Layout Height) = 150vh (50vh clean pause + 100vh overlap)
  // W (Wrapper Height) = 250vh (keeps element pinned perfectly during pause & overlap)
  // MB (Margin Bottom) = H - W = -100vh

  return (
    <div
      ref={outerRef}
      className="relative w-full"
      style={{
        height: '250vh',
        marginBottom: '-100vh',
        zIndex,
        perspective: '1200px',
        transformStyle: 'preserve-3d'
      }}
    >
      <div
        className={`sticky top-0 w-full ${bgClass}`}
        style={{ minHeight: '100svh' }}
      >
        <div
          ref={incomingRef}
          className="w-full h-full will-change-transform"
          style={{ transformOrigin: 'center bottom', minHeight: '100svh' }}
        >
          <div
            ref={contentRef}
            className="w-full h-full will-change-transform"
            style={{ transformOrigin: 'center top' }}
          >
            {/* The child content wrapper */}
            <div className="w-full h-full flex flex-col justify-center">
              {children}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
