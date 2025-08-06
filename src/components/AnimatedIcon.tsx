import React from 'react';
import { motion } from 'framer-motion';
import { DivideIcon as LucideIcon } from 'lucide-react';

interface AnimatedIconProps {
  icon: LucideIcon;
  className?: string;
  animation?: 'bounce' | 'spin' | 'pulse' | 'wiggle' | 'float';
  size?: number;
  color?: string;
}

export function AnimatedIcon({ 
  icon: Icon, 
  className = '', 
  animation = 'float',
  size = 24,
  color = 'currentColor'
}: AnimatedIconProps) {
  const animations = {
    bounce: {
      y: [0, -10, 0],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    },
    spin: {
      rotate: 360,
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "linear"
      }
    },
    pulse: {
      scale: [1, 1.2, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    },
    wiggle: {
      rotate: [0, -10, 10, -10, 0],
      transition: {
        duration: 0.5,
        repeat: Infinity,
        repeatDelay: 2
      }
    },
    float: {
      y: [0, -8, 0],
      x: [0, 4, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <motion.div
      className={className}
      animate={animations[animation]}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
    >
      <Icon size={size} color={color} />
    </motion.div>
  );
}

interface GlowingIconProps {
  icon: LucideIcon;
  className?: string;
  glowColor?: string;
  size?: number;
}

export function GlowingIcon({ 
  icon: Icon, 
  className = '', 
  glowColor = '#3b82f6',
  size = 24 
}: GlowingIconProps) {
  return (
    <motion.div
      className={`relative ${className}`}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
    >
      <motion.div
        className="absolute inset-0 rounded-full"
        animate={{
          boxShadow: [
            `0 0 20px ${glowColor}40`,
            `0 0 40px ${glowColor}60`,
            `0 0 20px ${glowColor}40`
          ]
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <Icon size={size} className="relative z-10" />
    </motion.div>
  );
}