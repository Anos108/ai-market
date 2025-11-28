-- Migration: Add initial_price tracking to managed_symbols
-- This allows us to track price changes from when symbols were added

-- Add columns if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='managed_symbols' AND column_name='initial_price') THEN
        ALTER TABLE managed_symbols ADD COLUMN initial_price DECIMAL(10,2);
        ALTER TABLE managed_symbols ADD COLUMN initial_price_date TIMESTAMP;
        
        -- Update existing records with NULL initial_price to fetch current price
        -- This will be populated by the application on next update
        RAISE NOTICE 'Added initial_price columns to managed_symbols table';
    ELSE
        RAISE NOTICE 'initial_price columns already exist in managed_symbols table';
    END IF;
END $$;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_managed_symbols_initial_price ON managed_symbols(initial_price) WHERE initial_price IS NOT NULL;

