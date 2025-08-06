import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { DivideIcon as LucideIcon } from 'lucide-react';

interface InteractiveCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  gradient: string;
  onClick?: () => void;
  children?: React.ReactNode;
  className?: string;
}

export function InteractiveCard({
  title,
  description,
  icon: Icon,
  gradient,
  onClick,
  children,
  className = ''
}: InteractiveCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      className={`interactive-card p-8 cursor-pointer ${className}`}
      onClick={onClick}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={{ 
        scale: 1.05,
        rotateY: 5,
        rotateX: 5,
      }}
      whileTap={{ scale: 0.95 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        type: "spring",
        stiffness: 300,
        damping: 20
      }}
    >
      {/* Animated background */}
      <motion.div
        className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${gradient} opacity-0`}
        animate={{ opacity: isHovered ? 0.1 : 0 }}
        transition={{ duration: 0.3 }}
      />
      
      {/* Glow effect */}
      <motion.div
        className="absolute inset-0 rounded-2xl"
        animate={{
          boxShadow: isHovered 
            ? '0 20px 40px rgba(59, 130, 246, 0.3)' 
            : '0 10px 20px rgba(0, 0, 0, 0.1)'
        }}
        transition={{ duration: 0.3 }}
      />

      <div className="relative z-10">
        {/* Icon */}
        <motion.div
          className={`w-16 h-16 bg-gradient-to-r ${gradient} rounded-2xl flex items-center justify-center mb-6 mx-auto`}
          animate={{ 
            rotate: isHovered ? 360 : 0,
            scale: isHovered ? 1.1 : 1
          }}
          transition={{ duration: 0.5 }}
        >
          <Icon className="w-8 h-8 text-white" />
        </motion.div>

        {/* Content */}
        <motion.h3 
          className="text-xl font-bold text-gray-900 dark:text-white mb-3 text-center"
          animate={{ y: isHovered ? -2 : 0 }}
          transition={{ duration: 0.2 }}
        >
          {title}
        </motion.h3>
        
        <motion.p 
          className="text-gray-600 dark:text-gray-300 text-center leading-relaxed"
          animate={{ y: isHovered ? -2 : 0 }}
          transition={{ duration: 0.2, delay: 0.1 }}
        >
          {description}
        </motion.p>

        {children && (
          <motion.div
            className="mt-6"
            animate={{ y: isHovered ? -2 : 0 }}
            transition={{ duration: 0.2, delay: 0.2 }}
          >
            {children}
          </motion.div>
        )}
      </div>

      {/* Particle effects */}
      {isHovered && (
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-blue-400 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ 
                scale: [0, 1, 0],
                opacity: [0, 1, 0],
                y: [0, -20],
              }}
              transition={{
                duration: 1,
                delay: i * 0.1,
                repeat: Infinity,
                repeatDelay: 2
              }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
}

interface MorphingCardProps {
  children: React.ReactNode;
  className?: string;
}

export function MorphingCard({ children, className = '' }: MorphingCardProps) {
  return (
    <motion.div
      className={`relative overflow-hidden rounded-2xl ${className}`}
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="absolute inset-0 morphing-bg opacity-20" />
      <div className="relative z-10 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl border border-gray-200/50 dark:border-gray-700/50">
        {children}
      </div>
    </motion.div>
  );
}