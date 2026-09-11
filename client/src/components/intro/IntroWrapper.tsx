'use client';

import React, { useState, useEffect } from 'react';
import { ShipRuleIntro } from './ShipRuleIntro';

interface IntroWrapperProps {
  children: React.ReactNode;
}

export const IntroWrapper: React.FC<IntroWrapperProps> = ({ children }) => {
  const [showIntro, setShowIntro] = useState<boolean>(true);
  const [isMounted, setIsMounted] = useState<boolean>(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handleIntroComplete = () => {
    setShowIntro(false);
  };

  if (!isMounted) {
    // Avoid SSR hydration flicker before mounting
    return <>{children}</>;
  }

  return (
    <>
      {showIntro && (
        <ShipRuleIntro onComplete={handleIntroComplete} />
      )}
      <div className={`transition-opacity duration-700 ${showIntro ? 'opacity-90' : 'opacity-100'}`}>
        {children}
      </div>
    </>
  );
};

export default IntroWrapper;
